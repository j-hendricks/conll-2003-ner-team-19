import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score
import numpy as np

from data import tokenizer, tokenized_dataset, id2label, num_labels, label2id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS      = [42, 123, 456]
BATCH_SIZE = 16

# drop unused columns
keep_cols = ["input_ids", "attention_mask", "labels"]
test_data = tokenized_dataset["test"].remove_columns(
    [c for c in tokenized_dataset["test"].column_names if c not in keep_cols]
)
test_data.set_format("torch")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
test_loader   = DataLoader(test_data, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=data_collator)


def evaluate_model(model_path, seed):
    # Load checkpoint for this seed
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch       = {k: v.to(device) for k, v in batch.items()}
            outputs     = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels      = batch["labels"]

            # Strip -100 padding and special tokens, convert ids to tag strings
            for pred_seq, label_seq in zip(predictions, labels):
                pred_tags = []
                true_tags = []
                for pred, lab in zip(pred_seq, label_seq):
                    if lab.item() == -100:
                        continue
                    pred_tags.append(id2label[pred.item()])
                    true_tags.append(id2label[lab.item()])
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    # Span-level F1 via seqeval
    f1 = f1_score(all_labels, all_preds)
    print(f"\nseed {seed}")
    print(f"  f1: {f1:.4f}")
    print(classification_report(all_labels, all_preds))
    return f1

#Run Eval 
if __name__ == "__main__":
    f1_scores = []
    for seed in SEEDS:
        f1 = evaluate_model(f"best_model_seed{seed}.pt", seed)
        f1_scores.append(f1)

    print("\nresults across 3 seeds")
    print(f" f1 per seed : {[round(f, 4) for f in f1_scores]}")
    print(f" mean f1 : {np.mean(f1_scores):.4f}")
    print(f" std f1 : {np.std(f1_scores):.4f}")