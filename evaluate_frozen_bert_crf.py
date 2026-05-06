import torch
import time
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

from data import tokenizer, tokenized_dataset, id2label, num_labels
from model_bert_crf import BertCRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456]
BATCH_SIZE = 16
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

keep_cols = ["input_ids", "attention_mask", "labels"]
test_data = tokenized_dataset["test"].remove_columns(
    [c for c in tokenized_dataset["test"].column_names if c not in keep_cols]
)
test_data.set_format("torch")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=data_collator)

NUM_TEST_SENTENCES = len(test_data)


def extract_spans(tag_seq):
    """Convert a BIO tag sequence into a set of (start, end, type) spans."""
    spans = set()
    start, etype = None, None
    for i, tag in enumerate(tag_seq):
        if tag.startswith("B-"):
            if start is not None:
                spans.add((start, i - 1, etype))
            start, etype = i, tag[2:]
        elif tag.startswith("I-"):
            if start is None or tag[2:] != etype:
                if start is not None:
                    spans.add((start, i - 1, etype))
                start, etype = i, tag[2:]
        else:  # O
            if start is not None:
                spans.add((start, i - 1, etype))
                start, etype = None, None
    if start is not None:
        spans.add((start, len(tag_seq) - 1, etype))
    return spans


def error_taxonomy(all_labels, all_preds):
    """
    4-category error breakdown across all sentences:
      a. wrong boundary  — same type, overlapping span but boundaries differ
      b. wrong type      — exact boundary match, wrong entity type
      c. missed entity   — gold span not predicted at all (false negative)
      d. hallucinated    — predicted span not in gold at all (false positive)
    """
    wrong_boundary = 0
    wrong_type = 0
    missed = 0
    hallucinated = 0

    for true_tags, pred_tags in zip(all_labels, all_preds):
        gold_spans = extract_spans(true_tags)
        pred_spans = extract_spans(pred_tags)

        gold_positions = {(s, e): t for s, e, t in gold_spans}
        pred_positions = {(s, e): t for s, e, t in pred_spans}

        gold_ranges = [(s, e, t) for s, e, t in gold_spans]
        pred_ranges = [(s, e, t) for s, e, t in pred_spans]

        matched_gold = set()
        matched_pred = set()

        # b. wrong type: exact boundary, wrong label
        for (s, e), gt in gold_positions.items():
            if (s, e) in pred_positions:
                matched_gold.add((s, e, gt))
                pt = pred_positions[(s, e)]
                matched_pred.add((s, e, pt))
                if gt != pt:
                    wrong_type += 1

        # a. wrong boundary: same type, overlapping span, boundaries differ
        for gs, ge, gt in gold_ranges:
            if (gs, ge, gt) in matched_gold:
                continue
            for ps, pe, pt in pred_ranges:
                if (ps, pe, pt) in matched_pred:
                    continue
                if gt == pt and gs <= pe and ps <= ge and (gs, ge) != (ps, pe):
                    wrong_boundary += 1
                    matched_gold.add((gs, ge, gt))
                    matched_pred.add((ps, pe, pt))
                    break

        # c. missed: gold spans with no match
        for span in gold_ranges:
            if span not in matched_gold:
                missed += 1

        # d. hallucinated: pred spans with no match
        for span in pred_ranges:
            if span not in matched_pred:
                hallucinated += 1

    total_errors = wrong_boundary + wrong_type + missed + hallucinated
    print("\n  -- error taxonomy --")
    print(f"  a. wrong boundary  : {wrong_boundary:4d}  ({100*wrong_boundary/max(total_errors,1):.1f}%)")
    print(f"  b. wrong type      : {wrong_type:4d}  ({100*wrong_type/max(total_errors,1):.1f}%)")
    print(f"  c. missed (FN)     : {missed:4d}  ({100*missed/max(total_errors,1):.1f}%)")
    print(f"  d. hallucinated(FP): {hallucinated:4d}  ({100*hallucinated/max(total_errors,1):.1f}%)")
    print(f"  total errors       : {total_errors}")
    return {"wrong_boundary": wrong_boundary, "wrong_type": wrong_type,
            "missed": missed, "hallucinated": hallucinated}


def per_entity_scores(all_labels, all_preds):
    """Compute precision, recall, F1 per entity type for one seed."""
    scores = {}
    for etype in ENTITY_TYPES:
        filtered_labels = []
        filtered_preds = []
        for true_seq, pred_seq in zip(all_labels, all_preds):
            true_filtered = [
                t if (t == f"B-{etype}" or t == f"I-{etype}") else "O"
                for t in true_seq
            ]
            pred_filtered = [
                p if (p == f"B-{etype}" or p == f"I-{etype}") else "O"
                for p in pred_seq
            ]
            filtered_labels.append(true_filtered)
            filtered_preds.append(pred_filtered)

        p = precision_score(filtered_labels, filtered_preds, zero_division=0)
        r = recall_score(filtered_labels, filtered_preds, zero_division=0)
        f = f1_score(filtered_labels, filtered_preds, zero_division=0)
        scores[etype] = {"precision": p, "recall": r, "f1": f}
    return scores


def evaluate_model(model_path, seed):
    model = BertCRF(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Warmup pass
    with torch.no_grad():
        warmup_batch = next(iter(test_loader))
        warmup_batch = {k: v.to(device) for k, v in warmup_batch.items()}
        _ = model(input_ids=warmup_batch["input_ids"],
                  attention_mask=warmup_batch["attention_mask"])
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            mask = batch["attention_mask"].bool()
            emissions = model(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"])
            predictions = model.crf.decode(emissions, mask=mask)

            for pred_seq, label_seq, mask_seq in zip(
                    predictions, batch["labels"], batch["attention_mask"]):
                pred_tags = []
                true_tags = []
                pred_idx = 0
                for lab, m in zip(label_seq, mask_seq):
                    if m.item() == 0:
                        continue
                    if lab.item() == -100:
                        pred_idx += 1
                        continue
                    pred_tags.append(id2label[pred_seq[pred_idx]])
                    true_tags.append(id2label[lab.item()])
                    pred_idx += 1
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - inference_start
    sents_per_sec = NUM_TEST_SENTENCES / elapsed

    f1 = f1_score(all_labels, all_preds)
    print(f"\nseed {seed}")
    print(f"  f1             : {f1:.4f}")
    print(f"  inference time : {elapsed:.2f}s over {NUM_TEST_SENTENCES} sentences")
    print(f"  throughput     : {sents_per_sec:.1f} sentences/sec")
    print(classification_report(all_labels, all_preds))
    error_taxonomy(all_labels, all_preds)

    entity_scores = per_entity_scores(all_labels, all_preds)
    return f1, sents_per_sec, entity_scores


def print_entity_summary(all_entity_scores):
    """Print mean ± std for precision, recall, F1 per entity type across seeds."""
    print("\n-- per-entity results across 3 seeds --")
    print(f"  {'entity':<6}  {'precision':>14}  {'recall':>14}  {'f1':>14}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*14}")
    for etype in ENTITY_TYPES:
        precs = [s[etype]["precision"] for s in all_entity_scores]
        recs = [s[etype]["recall"] for s in all_entity_scores]
        f1s = [s[etype]["f1"] for s in all_entity_scores]
        print(f"  {etype:<6}  "
              f"{np.mean(precs):.3f}±{np.std(precs):.3f}  "
              f"{np.mean(recs):.3f}±{np.std(recs):.3f}  "
              f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}")


if __name__ == "__main__":
    f1_scores = []
    speed_scores = []
    all_entity_scores = []

    for seed in SEEDS:
        f1, sps, entity_scores = evaluate_model(f"best_frozen_bertcrf_seed{seed}.pt", seed)
        f1_scores.append(f1)
        speed_scores.append(sps)
        all_entity_scores.append(entity_scores)

    print("\nresults across 3 seeds")
    print(f" f1 per seed        : {[round(f, 4) for f in f1_scores]}")
    print(f" mean f1            : {np.mean(f1_scores):.4f}")
    print(f" std f1             : {np.std(f1_scores):.4f}")
    print(f" mean throughput    : {np.mean(speed_scores):.1f} sents/sec")
    print(f" std throughput     : {np.std(speed_scores):.1f} sents/sec")
    print_entity_summary(all_entity_scores)