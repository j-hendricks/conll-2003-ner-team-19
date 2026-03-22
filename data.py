from datasets import DatasetDict, load_dataset
from transformers import BertTokenizerFast

BASE = "https://huggingface.co/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003"

dataset = DatasetDict({
    split: load_dataset("parquet", data_files={split: f"{BASE}/{split}/0000.parquet"}, split=split)
    for split in ["train", "validation", "test"]
})


# Model Label Mapping
ner_feature  = dataset["train"].features["ner_tags"] #Feature legend 
ner_labels = ner_feature.feature.names
# string key and int key dictionaries for tag - id conversations 
label2id     = {l: i for i, l in enumerate(ner_labels)} 
id2label     = {i: l for i, l in enumerate(ner_labels)}

num_labels   = len(ner_labels) #output classes 


#BERT Vocab prebuilt tokenizer 
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# Subword alignment for BERT splits
#Assign gold label to first sub-token only
# Rest are given -100 

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        max_length=200,
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) #ignore 
            elif word_idx != previous_word_idx:
                #Assign gold label 
                label_ids.append(label[word_idx])
            else:
                #Ignore 
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply to all splits
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

