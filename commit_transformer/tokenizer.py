# Define the Tokenizer
import torch
import re
from torch.nn.utils.rnn import pad_sequence
# Simple tokenizer function
def simple_tokenizer(text, vocab):
    tokens = re.findall(r'\b\w+\b', text.lower())
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return token_ids

# Build a vocabulary from the dataset
def build_vocab(data):
    vocab = {'<pad>': 0, '<unk>': 1}
    index = 2
    for sentence1, sentence2, _ in data:
        for token in re.findall(r'\b\w+\b', sentence1.lower()) + re.findall(r'\b\w+\b', sentence2.lower()):
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab


