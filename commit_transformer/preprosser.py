# Define the Dataset
import torch
from torch.utils.data import Dataset
from commit_transformer.tokenizer import simple_tokenizer
class CommitDataset(Dataset):
    def __init__(self, data, vocab, max_length=128):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def pad_and_truncate(self, token_ids):
        if len(token_ids) > self.max_length:
            return token_ids[:self.max_length]
        else:
            return token_ids + [self.vocab['<pad>']] * (self.max_length - len(token_ids))

    def get_embedding(self, idx):
        sentence1, sentence2, label = self.data[idx]
        token_ids1 = self.pad_and_truncate(simple_tokenizer(sentence1, self.vocab))
        token_ids2 = self.pad_and_truncate(simple_tokenizer(sentence2, self.vocab))
        return torch.tensor(token_ids1, dtype=torch.long), torch.tensor(token_ids2, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __getitem__(self, idx):
        return self.get_embedding(idx)