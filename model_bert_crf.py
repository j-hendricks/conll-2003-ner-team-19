import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class BertCRF(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_out = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_out)

        if labels is not None:
            # Use full attention mask — torchcrf requires the first timestep
            # ([CLS]) to always be unmasked. -100 positions are clamped to 0
            # so the CRF doesn't error; the alignment loop handles skipping them.
            mask = attention_mask.bool()
            clamped_labels = labels.clone()
            clamped_labels[labels == -100] = 0

            loss = -self.crf(emissions, clamped_labels, mask=mask, reduction="mean")
            return loss, emissions
        else:
            return emissions