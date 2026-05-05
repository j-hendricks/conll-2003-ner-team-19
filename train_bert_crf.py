import torch
import time
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, get_scheduler
from torch.optim import AdamW
from seqeval.metrics import f1_score as seqeval_f1

from data import tokenizer, tokenized_dataset, id2label, num_labels
from model_bert_crf import BertCRF

device = torch.device("cuda")

SEEDS = [42, 123, 456]
LR_BERT = 5e-5
LR_CRF = 1e-3
EPOCHS = 7
BATCH_SIZE = 64

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def train_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = BertCRF(num_labels=num_labels)
    model.to(device)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_data = tokenized_dataset["train"].remove_columns(
        [col for col in tokenized_dataset["train"].column_names if col not in keep_cols]
    )
    val_data = tokenized_dataset["validation"].remove_columns(
        [col for col in tokenized_dataset["validation"].column_names if col not in keep_cols]
    )
    train_data.set_format("torch")
    val_data.set_format("torch")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=data_collator,
                          num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=data_collator,
                          num_workers=4, pin_memory=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], "lr": LR_BERT, "weight_decay": 0.01},
        {"params": [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], "lr": LR_BERT, "weight_decay": 0.0},
        {"params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)], "lr": LR_CRF, "weight_decay": 0.01},
        {"params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)], "lr": LR_CRF, "weight_decay": 0.0},
        {"params": list(model.crf.parameters()), "lr": LR_CRF, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=warmup_steps,
                              num_training_steps=total_steps)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        start = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(**batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time = time.time() - start

        # Validation
        model.eval()
        val_preds  = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                # Full attention mask 
                mask = batch["attention_mask"].bool()
                emissions = model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"])
                predictions = model.crf.decode(emissions, mask=mask)

                # predictions
                for pred_seq, label_seq, mask_seq in zip(
                        predictions, batch["labels"], batch["attention_mask"]):
                    pred_tags = []
                    true_tags = []
                    pred_idx  = 0
                    for lab, m in zip(label_seq, mask_seq):
                        if m.item() == 0:
                            continue        # padding, not in pred_seq
                        if lab.item() == -100:
                            pred_idx += 1   # special/subword, skip but advance
                            continue
                        pred_tags.append(id2label[pred_seq[pred_idx]])
                        true_tags.append(id2label[lab.item()])
                        pred_idx += 1
                    val_preds.append(pred_tags)
                    val_labels.append(true_tags)

        val_f1 = seqeval_f1(val_labels, val_preds)

        print(f"  epoch {epoch+1}/{EPOCHS} | "
              f"train loss {avg_train_loss:.4f} | "
              f"val f1 {val_f1:.4f} | "
              f"{epoch_time:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"best_bertcrf_seed{seed}.pt")
            print(f"  saved (val f1 {best_val_f1:.4f})")


if __name__ == "__main__":
    for seed in SEEDS:
        print(f"\n=== BERT+CRF seed {seed} ===")
        train_seed(seed)
