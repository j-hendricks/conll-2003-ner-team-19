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
# This function makes results reproducible.
# It ensures that randomness (in Python, NumPy, and PyTorch)
# behaves the same every time we run the code with the same seed.
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. Load CoNLL-2003
# This section loads the CoNLL-2003 dataset.
# The dataset contains sentences with tokens and NER tags (BIO format).
# =========================
BASE = "https://huggingface.co/datasets/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003"

# Load dataset splits into a DatasetDict:
# - train
# - validation (dev)
# - test
dataset = DatasetDict({
    split: load_dataset("parquet", data_files={split: f"{BASE}/{split}/0000.parquet"}, split=split)
    for split in ["train", "validation", "test"]
})


# =========================
# 3. Special tokens
# Special tokens are used to handle edge cases in text processing.
# =========================

# <PAD> is used to pad sequences so that all sentences in a batch have the same length
PAD_TOKEN = "<PAD>"

# <UNK> represents unknown words that are not in the vocabulary.
# This happens when we see a word during validation/test that was not seen during training.
UNK_TOKEN = "<UNK>"

# Index for padding token (used when filling shorter sequences)
PAD_IDX = 0

# Index for padding token (used when filling shorter sequences)
UNK_IDX = 1


# =========================
# 4. Token normalization
# This function standardizes tokens before adding them to the vocabulary
# or feeding them into the model. This helps reduce sparsity and improve generalization.
# =========================
def normalize_token(token):

    # Convert token to lowercase
    token = token.lower()

    # Replace all digits with 0 (e.g., "1998" → "0000") to generalize numbers
    token = re.sub(r"\d", "0", token)
    return token


# =========================
# 5. Build vocabularies
# This function builds the word vocabulary from the dataset.
# A vocabulary is a dictionary that maps each normalized word to a unique integer index so the model can process text as numbers.
# =========================
def build_word_vocab(hf_dataset, min_freq=1):
    # Count how often each normalized word appears
    counter = Counter()
    for ex in hf_dataset:
        for token in ex["tokens"]:
            counter[normalize_token(token)] += 1

    # Start the vocabulary with special tokens for padding and unknown words
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}

    # Add words to the vocabulary if they appear at least min_freq times
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# This function builds a character vocabulary.
# It maps each character to a unique integer index.
def build_char_vocab(hf_dataset):
    # Start with special tokens for padding and unknown characters
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}

    # Go through every token and then every character in that token
    for ex in hf_dataset:
        for token in ex["tokens"]:
            for ch in token:
                # Add the character only if it is not already in the vocabulary
                if ch not in vocab:
                    vocab[ch] = len(vocab)
    return vocab

# Build the word vocabulary using only the training set
# so the model does not learn vocabulary information from validation/test directly
word_vocab = build_word_vocab(dataset["train"])

# Build the character vocabulary from the training set
char_vocab = build_char_vocab(dataset["train"])

# Get the list of NER label names from the dataset
# Example labels may include: O, B-PER, I-PER, B-LOC, I-LOC, etc.
label_names = dataset["train"].features["ner_tags"].feature.names

# Count how many unique NER tags there are
tagset_size = len(label_names)

# Create a mapping from integer tag ID back to tag name
# This is useful when converting predictions into readable BIO labels
id_to_tag = {i: tag for i, tag in enumerate(label_names)}


# =========================
# 6. Load GloVe
# This function loads pretrained GloVe embeddings and aligns them
# with the word vocabulary used by the model.
# Each word in the vocabulary gets a vector of size emb_dim.
# =========================
def load_glove_embeddings(glove_path, word_vocab, emb_dim=300):
    # Create an embedding matrix with random values for all words initially
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_vocab), emb_dim)).astype(np.float32)
    # Set the padding token embedding to all zeros
    embeddings[PAD_IDX] = np.zeros(emb_dim, dtype=np.float32)

    # Keep track of how many vocabulary words were found in GloVe
    found = 0

    # Open the GloVe text file and read it line by line
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            # Split each line into the word and its embedding values
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = parts[1:]

            if len(vec) != emb_dim:
                continue

            # If the word exists in our vocabulary, replace its random vector
            # with the pretrained GloVe vector
            if word in word_vocab:
                embeddings[word_vocab[word]] = np.asarray(vec, dtype=np.float32)
                found += 1

    print(f"GloVe matched: {found}/{len(word_vocab)} words")

     # Convert the NumPy array into a PyTorch tensor for use in the model
    return torch.tensor(embeddings, dtype=torch.float32)


# CHANGE THIS TO YOUR ACTUAL GLOVE FILE
# This should point to the actual location of glove.6B.300d.txt on your machine
GLOVE_PATH = "/home/ptm3ktr/Downloads/glove.6B.300d.txt"

# Load the GloVe vectors and create the embedding tensor
# This tensor will later initialize the word embedding layer in the model
glove_tensor = load_glove_embeddings(GLOVE_PATH, word_vocab, emb_dim=300)


# =========================
# 7. Dataset wrapper
# This class converts the raw Hugging Face dataset into a format
# that PyTorch can work with (Dataset object).
# It prepares word IDs, character IDs, and labels for each sentence.
# =========================
class CoNLLDataset(Dataset):
    # Initialize the dataset with:
    # - hf_dataset: raw dataset (train/validation/test)
    # - word_vocab: mapping from words → indices
    # - char_vocab: mapping from characters → indices
    def __init__(self, hf_dataset, word_vocab, char_vocab):
        self.dataset = hf_dataset
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab

    # Return the number of examples (sentences) in the dataset
    def __len__(self):
        return len(self.dataset)

    # Get a single example (sentence) by index
    def __getitem__(self, idx):
        ex = self.dataset[idx]

        # Extract tokens (words) and NER tags
        tokens = ex["tokens"]
        tags = ex["ner_tags"]

        # Convert each token into a word ID using the vocabulary
        # If a word is not found, use UNK_IDX
        word_ids = [self.word_vocab.get(normalize_token(tok), UNK_IDX) for tok in tokens]

        # Convert each token into a word ID using the vocabulary
        # If a word is not found, use UNK_IDX
        char_ids = [[self.char_vocab.get(ch, UNK_IDX) for ch in tok] for tok in tokens]

        # Return a dictionary containing all representations of the sentence
        return {
            "tokens": tokens,        # original words (for reference)
            "word_ids": word_ids,    # numerical representation of words
            "char_ids": char_ids,    # numerical representation of characters
            "tags": tags             # NER labels (BIO format)
        }


# Create dataset objects for each split

# Training data: used to learn model parameters
train_data = CoNLLDataset(dataset["train"], word_vocab, char_vocab)

# Validation (dev) data: used for tuning and early stopping
val_data = CoNLLDataset(dataset["validation"], word_vocab, char_vocab)

# Test data: used only for final evaluation
test_data = CoNLLDataset(dataset["test"], word_vocab, char_vocab)


# =========================
# 8. Collate function
# This function takes a list of examples from the Dataset
# and combines them into one batch.
# Since sentences have different lengths, it pads them so they can
# be stored in fixed-size tensors and processed together.
# =========================
def collate_fn(batch):
    # Number of sentences in this batch
    batch_size = len(batch)

    # Find the length of the longest sentence in the batch
    # This tells us how wide the word/tag tensors need to be
    max_seq_len = max(len(item["word_ids"]) for item in batch)

    # Find the length of the longest word (in characters) in the batch
    # This tells us how deep the character tensor needs to be
    max_word_len = max(len(chars) for item in batch for chars in item["char_ids"])

    # Create a padded tensor for word IDs
    # Shape: (batch_size, max_seq_len)
    # Fill with PAD_IDX so shorter sentences are padded
    word_ids = torch.full((batch_size, max_seq_len), PAD_IDX, dtype=torch.long)

    # Create a padded tensor for character IDs
    # Shape: (batch_size, max_seq_len, max_word_len)
    # Fill with PAD_IDX so shorter words/sentences are padded
    char_ids = torch.full((batch_size, max_seq_len, max_word_len), PAD_IDX, dtype=torch.long)

    # Create a padded tensor for NER tag IDs
    # Shape: (batch_size, max_seq_len)
    # Using 0 here is okay because padding positions will be ignored by the mask
    tags = torch.full((batch_size, max_seq_len), 0, dtype=torch.long)

    # Create a boolean mask to mark real tokens vs padded positions
    # True = real token
    # False = padding
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    # Fill the tensors example by example
    for i, item in enumerate(batch):
        # Length of the current sentence
        seq_len = len(item["word_ids"])

        # Copy the sentence's word IDs into the padded tensor
        word_ids[i, :seq_len] = torch.tensor(item["word_ids"], dtype=torch.long)

        # Copy the sentence's gold NER tags into the padded tensor
        tags[i, :seq_len] = torch.tensor(item["tags"], dtype=torch.long)

        # Mark the real token positions as True in the mask
        mask[i, :seq_len] = True

        # Copy each word's character IDs into the padded char tensor
        for j, chars in enumerate(item["char_ids"]):
            char_ids[i, j, :len(chars)] = torch.tensor(chars, dtype=torch.long)

    # Return the fully padded batch
    return {
        "word_ids": word_ids,  # padded word-level tensor
        "char_ids": char_ids,  # padded character-level tensor
        "tags": tags,          # padded gold labels
        "mask": mask           # identifies real tokens vs padding
    }


# Create DataLoaders to iterate through the dataset in batches

# Training loader:
# - uses shuffle=True so the model sees the training data in random order
# - uses collate_fn to pad variable-length sequences
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Validation loader:
# - shuffle=False because evaluation does not need random order
# - still uses collate_fn for padding
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Test loader:
# - shuffle=False for consistent final evaluation
# - still uses collate_fn for padding
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)


# =========================
# 9. Character LSTM
# This class builds a character-level representation for each word.
# Instead of only using whole-word embeddings, the model also looks at the characters inside each word.
# =========================
class CharLSTM(nn.Module):

    # Initialize the character embedding layer and the character-level LSTM
    # char_vocab_size defines how many unique characters are represented
    # char_emb_dim sets the size of each character embedding vector
    # char_hidden_dim defines the size of the character-level word representation produced by the LSTM
    # padding_idx ensures that padded characters are ignored during embedding.
    def __init__(self, char_vocab_size, char_emb_dim=30, char_hidden_dim=25, padding_idx=PAD_IDX):
        super().__init__()

        # Character embedding layer:
        # converts each character ID into a dense vector of size char_emb_dim
        self.char_embedding = nn.Embedding(
            char_vocab_size,
            char_emb_dim,
            padding_idx=padding_idx
        )

        # Character LSTM:
        # reads the sequence of character embeddings for each word
        # and produces a final hidden state summarizing that word
        self.char_lstm = nn.LSTM(
            input_size=char_emb_dim,      # size of each input character embedding
            hidden_size=char_hidden_dim,  # final character-based word representation size
            num_layers=1,                 # one-layer LSTM
            batch_first=True,             # input shape is (batch, seq, feature)
            bidirectional=False           # single-direction character LSTM
        )

    def forward(self, char_ids):
        # char_ids has shape (B, S, W)
        # B = batch size
        # S = number of tokens in each sentence
        # W = maximum number of characters in each word
        B, S, W = char_ids.size()

        # Flatten from (B, S, W) to (B*S, W)
        # This lets us treat every word in the batch as its own character sequence
        flat = char_ids.view(B * S, W)

        # Compute the true length of each word by counting non-padding characters
        # clamp(min=1) prevents zero-length sequences, which LSTM packing does not allow
        lengths = (flat != PAD_IDX).sum(dim=1).clamp(min=1)

        # Convert character IDs into character embeddings
        # Shape becomes (B*S, W, char_emb_dim)
        emb = self.char_embedding(flat)

        # Pack the padded character sequences so the LSTM ignores padding characters
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Run the packed character sequences through the LSTM
        # h_n is the final hidden state for each word
        _, (h_n, _) = self.char_lstm(packed)

        # Take the final hidden state from the last layer
        # Shape: (B*S, char_hidden_dim)
        char_rep = h_n[-1]

        # Reshape back to sentence format:
        # (B*S, char_hidden_dim) -> (B, S, char_hidden_dim)
        # Now each token in each sentence has a character-based representation
        char_rep = char_rep.view(B, S, -1)

        # Return character-level word representations
        return char_rep


# =========================
# 10. BiLSTM-CRF model
# This is the main NER model.
# It combines:
# 1. word embeddings (GloVe)
# 2. character-level word representations (CharLSTM)
# 3. a BiLSTM for sentence context
# 4. a CRF for structured sequence prediction
# =========================
class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        word_vocab_size,              # number of unique words in the word vocabulary
        char_vocab_size,              # number of unique characters in the character vocabulary
        tagset_size,                  # number of possible NER tags (O, B-PER, I-PER, etc.)
        word_emb_dim=300,             # size of each word embedding vector
        char_emb_dim=30,              # size of each character embedding vector
        char_hidden_dim=25,           # size of the final character-based word representation
        lstm_hidden_dim=256,          # hidden size of the BiLSTM
        lstm_layers=2,                # number of stacked BiLSTM layers
        dropout=0.5,                  # dropout probability for regularization
        pretrained_word_embeddings=None,  # optional pretrained GloVe embedding matrix
        freeze_word_embeddings=False      # if True, do not update word embeddings during training
    ):
        super().__init__()

        # Word embedding layer:
        # converts each word ID into a dense vector of size word_emb_dim
        self.word_embedding = nn.Embedding(
            word_vocab_size,
            word_emb_dim,
            padding_idx=PAD_IDX
        )

        # If pretrained GloVe embeddings are provided, copy them into the embedding layer
        if pretrained_word_embeddings is not None:
            self.word_embedding.weight.data.copy_(pretrained_word_embeddings)

        # Optionally freeze word embeddings so they do not change during training
        if freeze_word_embeddings:
            self.word_embedding.weight.requires_grad = False

        # Character encoder:
        # converts each token into a character-based vector using the CharLSTM
        self.char_encoder = CharLSTM(
            char_vocab_size=char_vocab_size,
            char_emb_dim=char_emb_dim,
            char_hidden_dim=char_hidden_dim,
            padding_idx=PAD_IDX
        )

        # Final token representation size = word embedding + character representation
        # 300 + 25 = 325
        combined_dim = word_emb_dim + char_hidden_dim

        # Dropout applied after combining word and character representations
        self.input_dropout = nn.Dropout(dropout)

        # Bidirectional LSTM:
        # reads the whole sentence left-to-right and right-to-left
        # so each token gets contextual information from both directions
        self.bilstm = nn.LSTM(
            input_size=combined_dim,     # size of each token input vector
            hidden_size=lstm_hidden_dim, # hidden size per direction
            num_layers=lstm_layers,      # stack 2 BiLSTM layers
            batch_first=True,            # input shape is (batch, seq, feature)
            bidirectional=True,          # forward + backward directions
            dropout=dropout              # dropout between LSTM layers
        )

        # Dropout after the BiLSTM output
        self.output_dropout = nn.Dropout(dropout)

        # Feedforward layer:
        # transforms the BiLSTM output before predicting tag scores
        self.ff = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim * 2)

        # Nonlinear activation after the feedforward layer
        self.act = nn.Tanh()

        # Final linear layer:
        # maps each contextual token representation to scores for each NER tag
        self.hidden2tag = nn.Linear(lstm_hidden_dim * 2, tagset_size)

        # CRF layer:
        # predicts the best tag sequence while modeling dependencies between tags
        self.crf = CRF(num_tags=tagset_size, batch_first=True)

    def get_emissions(self, word_ids, char_ids, mask):
        # Convert word IDs into word embedding vectors
        # Shape: (B, S) -> (B, S, 300)
        word_emb = self.word_embedding(word_ids)

        # Convert character IDs into character-based word vectors
        # Shape: (B, S, W) -> (B, S, 25)
        char_rep = self.char_encoder(char_ids)

        # Concatenate word and character representations
        # Shape: (B, S, 300) + (B, S, 25) -> (B, S, 325)
        x = torch.cat([word_emb, char_rep], dim=-1)

        # Apply dropout to the combined token representations
        x = self.input_dropout(x)

        # Compute true sentence lengths from the mask
        # These lengths are needed to pack the padded sequences
        lengths = mask.sum(dim=1).cpu()

        # Pack padded sequences so the BiLSTM ignores padding tokens
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # Run the packed sequences through the BiLSTM
        packed_out, _ = self.bilstm(packed)

        # Unpack back into padded tensor form
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True
        )

        # Apply dropout to the BiLSTM outputs
        lstm_out = self.output_dropout(lstm_out)

        # Apply feedforward layer
        h = self.ff(lstm_out)

        # Apply Tanh activation
        h = self.act(h)

        # Convert token representations into emission scores for each tag
        # Shape: (B, S, hidden*2) -> (B, S, tagset_size)
        emissions = self.hidden2tag(h)

        # Return emission scores for the CRF layer
        return emissions

    def forward(self, word_ids, char_ids, tags, mask):
        # Compute emission scores from the BiLSTM
        emissions = self.get_emissions(word_ids, char_ids, mask)

        # Compute the CRF log-likelihood of the gold tag sequence
        log_likelihood = self.crf(
            emissions,
            tags,
            mask=mask,
            reduction="mean"
        )

        # Return negative log-likelihood as the training loss
        # We negate it because we want to minimize loss
        return -log_likelihood

    def decode(self, word_ids, char_ids, mask):
        # Compute emission scores from the BiLSTM
        emissions = self.get_emissions(word_ids, char_ids, mask)

        # Use the CRF to decode the best tag sequence for each sentence
        return self.crf.decode(emissions, mask=mask)


# =========================
# 11. Evaluation
# This function evaluates the model on a dataset (validation or test).
# It computes the F1 score using seqeval, which measures
# entity-level performance (not token-level accuracy).
# =========================
def evaluate(model, data_loader, id_to_tag, device):

    # Set the model to evaluation mode
    # This disables dropout and other training-specific behavior
    model.eval()

    # Lists to store all true and predicted tag sequences
    all_true = []
    all_pred = []

    # Disable gradient computation (faster and uses less memory)
    with torch.no_grad():

        # Loop over batches in the dataset
        for batch in data_loader:

            # Move batch data to device (CPU or GPU)
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            # Decode predicted tag sequences using the CRF
            # This returns the best tag sequence for each sentence
            pred_paths = model.decode(word_ids, char_ids, mask)

            # Loop through each sentence in the batch
            for i in range(word_ids.size(0)):

                # Compute the true length of the sentence (ignoring padding)
                seq_len = int(mask[i].sum().item())

                # Convert true tag IDs into tag names (BIO format)
                true_seq = [
                    id_to_tag[int(t.item())]
                    for t in tags[i, :seq_len]
                ]

                # Convert predicted tag IDs into tag names
                pred_seq = [
                    id_to_tag[int(t)]
                    for t in pred_paths[i]
                ]

                # Store the sequences for evaluation
                all_true.append(true_seq)
                all_pred.append(pred_seq)

    # Compute entity-level F1 score using seqeval
    f1 = f1_score(all_true, all_pred)

    # Generate a detailed classification report (precision, recall, F1 per class)
    report = classification_report(all_true, all_pred)

    # Return both the F1 score and the detailed report
    return f1, report


# =========================
# 12. Run one full experiment
# =========================

# Choose the device to run the model on.
# If a GPU is available, use it for faster training.
# Otherwise, use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# This function runs one complete training + evaluation experiment
# for a single random seed.
def run_experiment(seed):
    # Print the seed being used for this run
    print(f"\nRunning with seed {seed}")

    # Set all random seeds so the run is reproducible
    set_seed(seed)

    # Create a fresh BiLSTM-CRF model for this seed
    model = BiLSTMCRF(
        word_vocab_size=len(word_vocab),      # number of unique words
        char_vocab_size=len(char_vocab),      # number of unique characters
        tagset_size=tagset_size,              # number of output NER tags
        word_emb_dim=300,                     # GloVe embedding size
        char_emb_dim=30,                      # character embedding size
        char_hidden_dim=25,                   # character LSTM output size
        lstm_hidden_dim=256,                  # BiLSTM hidden size
        lstm_layers=2,                        # number of BiLSTM layers
        dropout=0.5,                          # dropout probability
        pretrained_word_embeddings=glove_tensor,  # initialize word embeddings with GloVe
        freeze_word_embeddings=False          # allow GloVe embeddings to update during training
    ).to(device)  # move the model to CPU or GPU

    optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

    # Track the best validation (dev) F1 score seen so far
    best_dev_f1 = 0.0

    # Store the model parameters for the best validation score
    best_state = None

    # Patience controls early stopping:
    # if validation F1 does not improve for 10 epochs in a row, stop training
    patience = 10
    patience_counter = 0

    # Maximum number of epochs to train
    num_epochs = 100

    # Training loop over epochs
    for epoch in range(1, num_epochs + 1):
        # Set model to training mode
        # This enables dropout and gradient updates
        model.train()

        # Track total loss across all batches in this epoch
        total_loss = 0.0

        # Loop over training batches
        for batch in train_loader:
            # Move batch data to the selected device
            word_ids = batch["word_ids"].to(device)
            char_ids = batch["char_ids"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            # Reset gradients from the previous batch
            optimizer.zero_grad()

            # Compute the loss for this batch
            # This calls the model's forward() function
            loss = model(word_ids, char_ids, tags, mask)

            # Compute gradients using backpropagation
            loss.backward()

            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Update model parameters using SGD
            optimizer.step()

            # Add the batch loss to the epoch total
            total_loss += loss.item()

        # Compute average training loss for this epoch
        avg_loss = total_loss / len(train_loader)

        # Evaluate the model on the validation (dev) set
        dev_f1, _ = evaluate(model, val_loader, id_to_tag, device)

        # Print training progress for this epoch
        print(f"Seed {seed} | Epoch {epoch:02d} | loss={avg_loss:.4f} | dev_f1={dev_f1:.4f}")

        # If validation F1 improved, save this as the best model so far
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1

            # Save a copy of the model weights
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            # Reset patience counter because performance improved
            patience_counter = 0
            print("New best model.")
        else:
            # If validation F1 did not improve, increase patience counter
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience}).")

        # Stop training early if there has been no improvement for too long
        if patience_counter >= patience:
            print("Early stopping.")
            break

    # Restore the best model weights before testing
    model.load_state_dict(best_state)

    # Evaluate the best model on the test set
    test_f1, test_report = evaluate(model, test_loader, id_to_tag, device)

    # Print final results for this seed
    print(f"\nSeed {seed} results:")
    print(f"Best dev F1: {best_dev_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(test_report)

    # Return the best dev F1 and final test F1 for this seed
    return best_dev_f1, test_f1


# =========================
# 13. Run all 3 seeds
# =========================
seeds = [42, 123, 456]

dev_scores = []
test_scores = []

for seed in seeds:
    dev_f1, test_f1 = run_experiment(seed)
    dev_scores.append(dev_f1)
    test_scores.append(test_f1)


# =========================
# 14. Mean ± std
# =========================
dev_mean = np.mean(dev_scores)
dev_std = np.std(dev_scores)

test_mean = np.mean(test_scores)
test_std = np.std(test_scores)

print("\n=========================")
print("Final Results Across Seeds")
print("=========================")
print(f"Seeds: {seeds}")
print(f"Dev F1:  {dev_mean:.4f} ± {dev_std:.4f}")
print(f"Test F1: {test_mean:.4f} ± {test_std:.4f}")