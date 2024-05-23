import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AdamW
from commit_transformer.tokenizer import simple_tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(TransformerEncoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(128, embed_dim))  # Max length is 128
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:x.size(1), :]
        x = self.transformer_encoder(x)
        return x.mean(dim=1)  # Mean pooling over sequence length

class CombinedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout):
        super(CombinedModel, self).__init__()
        self.encoder1 = TransformerEncoderModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.encoder2 = TransformerEncoderModel(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim * 2, 2)

    def forward(self, inputs1, inputs2):
        outputs1 = self.encoder1(inputs1)
        outputs2 = self.encoder2(inputs2)
        combined = torch.cat((outputs1, outputs2), dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

    def predict(self, sentence1, sentence2, vocab):
        self.eval()
        with torch.no_grad():
            inputs1 = torch.tensor([simple_tokenizer(sentence1, vocab)], dtype=torch.long).to(next(self.parameters()).device)
            inputs2 = torch.tensor([simple_tokenizer(sentence2, vocab)], dtype=torch.long).to(next(self.parameters()).device)
            inputs1 = inputs1[:, :128]  # Ensure inputs are of max_length 128
            inputs2 = inputs2[:, :128]  # Ensure inputs are of max_length 128
            logits = self.forward(inputs1, inputs2)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            return predicted_class, probabilities.cpu().numpy()

    def trainer(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                inputs1, inputs2, labels = batch
                inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = self(inputs1, inputs2)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

        self.evaluate(val_loader)

    def evaluate(self, val_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for batch in val_loader:
                inputs1, inputs2, labels = batch
                inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

                logits = self(inputs1, inputs2)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = correct / total
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f'Validation Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')