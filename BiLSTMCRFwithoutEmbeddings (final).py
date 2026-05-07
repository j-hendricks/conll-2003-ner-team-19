import random
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from collections import Counter
from datasets import DatasetDict, load_dataset
from torchcrf import CRF
from seqeval.metrics import f1_score, classification_report


# =========================
# 1. Reproducibility
# =========================
def set_seed(seed):
    """
    Set random seeds for reproducibility.

    Ensures consistent behavior across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. Load CoNLL-2003
# =========================
BASE = "https://huggingface.co/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003"

dataset = DatasetDict({
    split: load_dataset(
        "parquet",
        data_files={split: f"{BASE}/{split}/0000.parquet"},
        split=split
    )
    for split in ["train", "validation", "test"]
})


# =========================
# 3. Special tokens
# =========================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1


# =========================
# 4. Token normalization
# =========================
def normalize_token(token):
    """
    Normalize a token for vocabulary consistency.

    Converts to lowercase and replaces digits with 0.

    Args:
        token (str): Input token.

    Returns:
        str: Normalized token.
    """
    token = token.lower()
    token = re.sub(r"\d", "0", token)
    return token


# =========================
# 5. GloVe vocab loader
# =========================
def load_glove_vocab(glove_path):
    """
    Load the set of words that appear in the GloVe file.

    Args:
        glove_path (str): Path to GloVe text file.

    Returns:
        set: Set of words found in GloVe.
    """
    glove_words = set()

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")

            if len(parts) < 10:
                continue

            word = parts[0]
            glove_words.add(word)

    return glove_words


# =========================
# 6. Build vocabularies
# =========================
def build_word_vocab(train_dataset, dev_dataset, test_dataset, glove_words, min_freq=1):
    """
    Build a word vocabulary.

    All training words above the minimum frequency are included.
    Words from dev/test are also added if they appear in GloVe.

    Args:
        train_dataset: Training split.
        dev_dataset: Validation split.
        test_dataset: Test split.
        glove_words (set): Words available in GloVe.
        min_freq (int): Minimum frequency for training words.

    Returns:
        dict: Mapping from word to index.
    """
    counter = Counter()

    for ex in train_dataset:
        for token in ex["tokens"]:
            counter[normalize_token(token)] += 1

    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    # Add dev/test words only if GloVe has them.
    # This improves word coverage at evaluation time without using labels.
    for split_dataset in [dev_dataset, test_dataset]:
        for ex in split_dataset:
            for token in ex["tokens"]:
                word = normalize_token(token)

                if word not in vocab and word in glove_words:
                    vocab[word] = len(vocab)

    return vocab


# =========================
# 7. Load GloVe and build vocab
# =========================
GLOVE_PATH = "/home/ptm3ktr/Downloads/glove.6B.300d.txt"

print("Loading GloVe vocabulary...")
glove_words = load_glove_vocab(GLOVE_PATH)

word_vocab = build_word_vocab(
    dataset["train"],
    dataset["validation"],
    dataset["test"],
    glove_words,
    min_freq=1
)

label_names = dataset["train"].features["ner_tags"].feature.names
tagset_size = len(label_names)
id_to_tag = {i: tag for i, tag in enumerate(label_names)}


# =========================
# 8. Load GloVe embeddings
# =========================
def load_glove_embeddings(glove_path, word_vocab, emb_dim=300):
    """
    Load pretrained GloVe vectors aligned to the model vocabulary.

    Args:
        glove_path (str): Path to GloVe file.
        word_vocab (dict): Word-to-index vocabulary.
        emb_dim (int): Embedding dimension.

    Returns:
        torch.Tensor: Embedding matrix of shape (vocab_size, emb_dim).
    """
    embeddings = np.random.uniform(
        -0.25,
        0.25,
        (len(word_vocab), emb_dim)
    ).astype(np.float32)

    embeddings[PAD_IDX] = np.zeros(emb_dim, dtype=np.float32)

    found = 0

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = parts[1:]

            if len(vec) != emb_dim:
                continue

            if word in word_vocab:
                embeddings[word_vocab[word]] = np.asarray(vec, dtype=np.float32)
                found += 1

    print(f"GloVe matched: {found}/{len(word_vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)


glove_tensor = load_glove_embeddings(GLOVE_PATH, word_vocab, emb_dim=300)


# =========================
# 9. Dataset wrapper
# =========================
class CoNLLDataset(Dataset):
    """
    PyTorch Dataset wrapper for CoNLL-2003.

    For this no-character ablation, examples are converted into:
    - original tokens
    - word IDs
    - gold NER tags

    Character IDs and capitalization IDs are intentionally removed
    because this ablation uses only GloVe word embeddings.
    """

    def __init__(self, hf_dataset, word_vocab):
        """
        Initialize the dataset wrapper.

        Args:
            hf_dataset: Hugging Face dataset split.
            word_vocab (dict): Mapping from word to index.
        """
        self.dataset = hf_dataset
        self.word_vocab = word_vocab

    def __len__(self):
        """
        Return the number of examples.

        Returns:
            int: Number of sentences.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve one example and convert it to numerical form.

        Args:
            idx (int): Example index.

        Returns:
            dict: Tokens, word_ids, and tags.
        """
        ex = self.dataset[idx]
        tokens = ex["tokens"]
        tags = ex["ner_tags"]

        word_ids = [
            self.word_vocab.get(normalize_token(tok), UNK_IDX)
            for tok in tokens
        ]

        return {
            "tokens": tokens,
            "word_ids": word_ids,
            "tags": tags
        }


train_data = CoNLLDataset(dataset["train"], word_vocab)
val_data = CoNLLDataset(dataset["validation"], word_vocab)
test_data = CoNLLDataset(dataset["test"], word_vocab)


# =========================
# 10. Collate function
# =========================
def collate_fn(batch):
    """
    Combine a list of examples into a padded batch.

    Pads variable-length sentences and creates tensors for:
    - word IDs
    - gold tags
    - mask

    Args:
        batch (list): List of dataset items.

    Returns:
        dict: Batched tensors ready for model input.
    """
    batch_size = len(batch)

    max_seq_len = max(
        len(item["word_ids"])
        for item in batch
    )

    word_ids = torch.full(
        (batch_size, max_seq_len),
        PAD_IDX,
        dtype=torch.long
    )

    tags = torch.full(
        (batch_size, max_seq_len),
        0,
        dtype=torch.long
    )

    mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.bool
    )

    for i, item in enumerate(batch):
        seq_len = len(item["word_ids"])

        word_ids[i, :seq_len] = torch.tensor(item["word_ids"], dtype=torch.long)
        tags[i, :seq_len] = torch.tensor(item["tags"], dtype=torch.long)
        mask[i, :seq_len] = True

    return {
        "word_ids": word_ids,
        "tags": tags,
        "mask": mask
    }


# =========================
# 11. Ablation model: BiLSTM-CRF without char embeddings
# =========================
class BiLSTMCRFNoChar(nn.Module):
    """
    BiLSTM-CRF ablation model without character embeddings.

    This model removes:
    - character BiLSTM representations
    - capitalization embeddings

    It uses only:
    - pretrained GloVe word embeddings
    - sentence-level BiLSTM
    - CRF decoding

    Architecture:
        GloVe word embeddings -> BiLSTM -> Linear layer -> CRF
    """

    def __init__(
        self,
        word_vocab_size,
        tagset_size,
        word_emb_dim=300,
        lstm_hidden_dim=256,
        lstm_layers=2,
        dropout=0.5,
        pretrained_word_embeddings=None,
        freeze_word_embeddings=False
    ):
        """
        Initialize the GloVe-only BiLSTM-CRF model.

        Args:
            word_vocab_size (int): Number of unique words.
            tagset_size (int): Number of NER output tags.
            word_emb_dim (int): Word embedding size.
            lstm_hidden_dim (int): BiLSTM hidden size per direction.
            lstm_layers (int): Number of BiLSTM layers.
            dropout (float): Dropout probability.
            pretrained_word_embeddings (torch.Tensor, optional): GloVe embedding matrix.
            freeze_word_embeddings (bool): Whether to freeze word embeddings.
        """
        super().__init__()

        self.word_embedding = nn.Embedding(
            word_vocab_size,
            word_emb_dim,
            padding_idx=PAD_IDX
        )

        if pretrained_word_embeddings is not None:
            self.word_embedding.weight.data.copy_(pretrained_word_embeddings)

        if freeze_word_embeddings:
            self.word_embedding.weight.requires_grad = False

        # Since character and capitalization features are removed,
        # the BiLSTM input is only the 300-dimensional GloVe word embedding.
        combined_dim = word_emb_dim

        self.input_dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.output_dropout = nn.Dropout(dropout)

        self.ff = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim * 2)
        self.act = nn.Tanh()

        self.hidden2tag = nn.Linear(lstm_hidden_dim * 2, tagset_size)

        self.crf = CRF(
            num_tags=tagset_size,
            batch_first=True
        )

    def get_emissions(self, word_ids, mask):
        """
        Compute emission scores using only word embeddings.

        Args:
            word_ids (torch.Tensor): Word ID tensor of shape (B, S).
            mask (torch.Tensor): Boolean mask tensor of shape (B, S).

        Returns:
            torch.Tensor: Emission scores of shape (B, S, tagset_size).
        """
        x = self.word_embedding(word_ids)
        x = self.input_dropout(x)

        lengths = mask.sum(dim=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.bilstm(packed)

        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        lstm_out = self.output_dropout(lstm_out)

        h = self.ff(lstm_out)
        h = self.act(h)

        emissions = self.hidden2tag(h)

        return emissions

    def forward(self, word_ids, tags, mask):
        """
        Compute CRF negative log-likelihood loss.

        Args:
            word_ids (torch.Tensor): Word ID tensor.
            tags (torch.Tensor): Gold NER tag tensor.
            mask (torch.Tensor): Boolean mask tensor.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        emissions = self.get_emissions(word_ids, mask)

        log_likelihood = self.crf(
            emissions,
            tags,
            mask=mask,
            reduction="mean"
        )

        return -log_likelihood

    def decode(self, word_ids, mask):
        """
        Decode the best tag sequence using the CRF.

        Args:
            word_ids (torch.Tensor): Word ID tensor.
            mask (torch.Tensor): Boolean mask tensor.

        Returns:
            list: Predicted tag sequences.
        """
        emissions = self.get_emissions(word_ids, mask)
        return self.crf.decode(emissions, mask=mask)


# =========================
# 12. Error taxonomy helpers
# =========================
def extract_entities(tags):
    """
    Convert a BIO tag sequence into entity spans.

    Each entity is represented as:
        (start_index, end_index, entity_type)

    The end index is exclusive.

    Example:
        ["B-PER", "I-PER", "O", "B-LOC"]
        -> [(0, 2, "PER"), (3, 4, "LOC")]

    Args:
        tags (list[str]): BIO tag sequence.

    Returns:
        list[tuple]: Entity spans.
    """
    entities = []
    start = None
    ent_type = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if start is not None:
                entities.append((start, i, ent_type))
                start = None
                ent_type = None
            continue

        if "-" not in tag:
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            if start is not None:
                entities.append((start, i, ent_type))

            start = i
            ent_type = entity_type

        elif prefix == "I":
            if start is None:
                start = i
                ent_type = entity_type
            elif entity_type != ent_type:
                entities.append((start, i, ent_type))
                start = i
                ent_type = entity_type

    if start is not None:
        entities.append((start, len(tags), ent_type))

    return entities


def entity_overlap(ent1, ent2):
    """
    Check whether two entity spans overlap.

    Args:
        ent1 (tuple): (start, end, type)
        ent2 (tuple): (start, end, type)

    Returns:
        bool: True if the spans overlap.
    """
    start1, end1, _ = ent1
    start2, end2, _ = ent2

    return start1 < end2 and start2 < end1


def compute_error_taxonomy(all_true, all_pred):
    """
    Compute NER error taxonomy counts.

    Error categories:
        correct_type_wrong_boundary:
            Prediction overlaps a gold entity and has the correct entity type,
            but the span boundaries are wrong.

        correct_boundary_wrong_type:
            Prediction has the same span as a gold entity,
            but the entity type is wrong.

        missed_entity:
            A gold entity is not matched by any prediction.

        hallucinated_entity:
            A predicted entity is not matched by any gold entity.

    Args:
        all_true (list[list[str]]): Gold BIO tag sequences.
        all_pred (list[list[str]]): Predicted BIO tag sequences.

    Returns:
        dict: Error counts by category.
    """
    errors = {
        "correct_type_wrong_boundary": 0,
        "correct_boundary_wrong_type": 0,
        "missed_entity": 0,
        "hallucinated_entity": 0
    }

    for true_tags, pred_tags in zip(all_true, all_pred):
        gold_entities = extract_entities(true_tags)
        pred_entities = extract_entities(pred_tags)

        matched_gold = set()
        matched_pred = set()

        # Exact matches are correct and should not be counted as errors.
        for gi, gold in enumerate(gold_entities):
            for pi, pred in enumerate(pred_entities):
                if gold == pred:
                    matched_gold.add(gi)
                    matched_pred.add(pi)

        # Correct boundary but wrong type.
        for gi, gold in enumerate(gold_entities):
            if gi in matched_gold:
                continue

            g_start, g_end, g_type = gold

            for pi, pred in enumerate(pred_entities):
                if pi in matched_pred:
                    continue

                p_start, p_end, p_type = pred

                if g_start == p_start and g_end == p_end and g_type != p_type:
                    errors["correct_boundary_wrong_type"] += 1
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        # Correct type but wrong boundary.
        for gi, gold in enumerate(gold_entities):
            if gi in matched_gold:
                continue

            g_start, g_end, g_type = gold

            for pi, pred in enumerate(pred_entities):
                if pi in matched_pred:
                    continue

                p_start, p_end, p_type = pred

                if g_type == p_type and entity_overlap(gold, pred):
                    errors["correct_type_wrong_boundary"] += 1
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        # Remaining unmatched gold entities are missed.
        for gi in range(len(gold_entities)):
            if gi not in matched_gold:
                errors["missed_entity"] += 1

        # Remaining unmatched predictions are hallucinated.
        for pi in range(len(pred_entities)):
            if pi not in matched_pred:
                errors["hallucinated_entity"] += 1

    return errors


# =========================
# 13. Evaluation
# =========================
def evaluate(model, data_loader, id_to_tag, device):
    """
    Evaluate the model on a dataset split.

    Uses seqeval to compute entity-level F1 and produces
    an error taxonomy for model analysis.

    Args:
        model: Trained model.
        data_loader: Validation or test DataLoader.
        id_to_tag (dict): Mapping from tag ID to tag name.
        device: CPU or GPU device.

    Returns:
        tuple: (f1_score, classification_report, error_taxonomy)
    """
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in data_loader:
            word_ids = batch["word_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            pred_paths = model.decode(word_ids, mask)

            for i in range(word_ids.size(0)):
                seq_len = int(mask[i].sum().item())

                true_seq = [
                    id_to_tag[int(t.item())]
                    for t in tags[i, :seq_len]
                ]

                pred_seq = [
                    id_to_tag[int(t)]
                    for t in pred_paths[i]
                ]

                all_true.append(true_seq)
                all_pred.append(pred_seq)

    f1 = f1_score(all_true, all_pred)
    report = classification_report(all_true, all_pred)
    errors = compute_error_taxonomy(all_true, all_pred)

    return f1, report, errors


# =========================
# 14. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# 15. Run no-character ablation
# =========================
def run_no_char_ablation(seed):
    """
    Run the BiLSTM-CRF without character embeddings ablation for one seed.

    This model uses only pretrained GloVe word embeddings as input
    to the sentence-level BiLSTM and keeps the CRF output layer.

    Args:
        seed (int): Random seed.

    Returns:
        tuple: (best_dev_f1, test_f1, test_errors)
    """
    print(f"\nRunning BiLSTM-CRF no-character ablation with seed {seed}")
    set_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BiLSTMCRFNoChar(
        word_vocab_size=len(word_vocab),
        tagset_size=tagset_size,
        word_emb_dim=300,
        lstm_hidden_dim=256,
        lstm_layers=2,
        dropout=0.5,
        pretrained_word_embeddings=glove_tensor,
        freeze_word_embeddings=False
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

    best_dev_f1 = 0.0
    best_state = None

    patience = 10
    patience_counter = 0
    num_epochs = 100

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            word_ids = batch["word_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            loss = model(
                word_ids,
                tags,
                mask
            )

            loss.backward()

            clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        dev_f1, _, _ = evaluate(
            model,
            val_loader,
            id_to_tag,
            device
        )

        print(
            f"No-Char Ablation | Seed {seed} | Epoch {epoch:02d} | "
            f"loss={avg_loss:.4f} | dev_f1={dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0
            print("New best model.")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience}).")

        if patience_counter >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)

    test_f1, test_report, test_errors = evaluate(
        model,
        test_loader,
        id_to_tag,
        device
    )

    print(f"\nBiLSTM-CRF no-character ablation | Seed {seed} results:")
    print(f"Best dev F1: {best_dev_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(test_report)

    print("Error taxonomy:")
    for error_type, count in test_errors.items():
        print(f"{error_type}: {count}")

    return best_dev_f1, test_f1, test_errors


# =========================
# 16. Run no-character ablation across 3 seeds
# =========================
seeds = [123, 456]

dev_scores = []
test_scores = []

error_totals = {
    "correct_type_wrong_boundary": [],
    "correct_boundary_wrong_type": [],
    "missed_entity": [],
    "hallucinated_entity": []
}

for seed in seeds:
    dev_f1, test_f1, test_errors = run_no_char_ablation(seed)

    dev_scores.append(dev_f1)
    test_scores.append(test_f1)

    for error_type in error_totals:
        error_totals[error_type].append(test_errors[error_type])


dev_mean = np.mean(dev_scores)
dev_std = np.std(dev_scores)

test_mean = np.mean(test_scores)
test_std = np.std(test_scores)


# =========================
# 17. Final results
# =========================
print("\n=========================")
print("BiLSTM-CRF No-Character Ablation Results Across Seeds")
print("=========================")
print(f"Seeds: {seeds}")
print(f"Dev F1:  {dev_mean:.4f} ± {dev_std:.4f}")
print(f"Test F1: {test_mean:.4f} ± {test_std:.4f}")

print("\nError Taxonomy Across Seeds")
for error_type, counts in error_totals.items():
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    print(f"{error_type}: {mean_count:.2f} ± {std_count:.2f}")