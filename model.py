from transformers import BertForTokenClassification
from data import label2id, id2label, num_labels


model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=num_labels,   
    id2label=id2label,       
    label2id=label2id        
)