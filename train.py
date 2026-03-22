import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, get_scheduler
from torch.optim import AdamW
from seqeval.metrics import f1_score as seqeval_f1
import time

from data import tokenizer, tokenized_dataset

device = torch.device("cuda")

# proposal configurations 
SEEDS      = [42, 123, 456]
LR         = 5e-5
EPOCHS     = 7
BATCH_SIZE = 16



data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def train_seed(seed):

    # reset for each seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from transformers import BertForTokenClassification
    from data import label2id, id2label, num_labels

    seed_model = BertForTokenClassification.from_pretrained(
        "bert-large-cased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    seed_model.to(device)

    # Drop columns BERT doesn't need
    keep_cols  = ["input_ids", "attention_mask", "labels"]
    train_data = tokenized_dataset["train"].remove_columns(
        [col for col in tokenized_dataset["train"].column_names if col not in keep_cols]
    )
    val_data   = tokenized_dataset["validation"].remove_columns(
        [col for col in tokenized_dataset["validation"].column_names if col not in keep_cols]
    )
    train_data.set_format("torch")
    val_data.set_format("torch")

    # Batch iterators
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=data_collator)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=data_collator)

    #Added Weight DEcay 
    no_decay  = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{"params": [p for n, p in seed_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in seed_model.named_parameters() if any(nd in n for nd in no_decay)],  "weight_decay": 0.0},
    ]

    # AdamW with linear warmup over first 10% of steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_scheduler("linear",optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Starting f1
    best_val_f1 = 0.0
    # best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        #Train 
        seed_model.train()
        total_train_loss = 0
        start = time.time()

        for batch in train_loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = seed_model(**batch)
            loss    = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(seed_model.parameters(), max_norm=1.0)  # prevent exploding gradients
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        epoch_time     = time.time() - start

        # Validate based on F1 
        seed_model.eval()
        total_val_loss = 0
        val_preds      = []
        val_labels     = []

        with torch.no_grad():
            for batch in val_loader:
                batch   = {k: v.to(device) for k, v in batch.items()}
                outputs = seed_model(**batch)
                total_val_loss += outputs.loss.item()

                # Collect predictions for entity-level F1
                predictions = outputs.logits.argmax(dim=-1)
                for pred_seq, label_seq in zip(predictions, batch["labels"]):
                    pred_tags = []
                    true_tags = []
                    for p, l in zip(pred_seq, label_seq):
                        if l.item() == -100:
                            # Skip padding and special tokens
                            continue
                        pred_tags.append(id2label[p.item()])
                        true_tags.append(id2label[l.item()])
                    val_preds.append(pred_tags)
                    val_labels.append(true_tags)

        avg_val_loss = total_val_loss / len(val_loader)

        val_f1 = seqeval_f1(val_labels, val_preds)

        # Epoch summary
        print(f"  epoch {epoch+1}/{EPOCHS} | "
              f"train loss {avg_train_loss:.4f} | "
              f"val loss {avg_val_loss:.4f} | "
              f"val f1 {val_f1:.4f} | "
              f"{epoch_time:.1f}s")

        # Save best checkpoint by val F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(seed_model.state_dict(), f"best_model_seed{seed}.pt")
            print(f"  saved (val f1 {best_val_f1:.4f})")

        ''' 
        #Old loss-based saving f1 avg @ .90 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(seed_model.state_dict(), f"best_model_seed{seed}.pt")
            print(f" Saved best val loss {best_val_loss:.4f}")
        '''
    return seed_model


# Run all 3 seeds
if __name__ == "__main__":
    trained_models = {}
    for seed in SEEDS:
        trained_models[seed] = train_seed(seed)