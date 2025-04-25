import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configs
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset
df = pd.read_csv("spam_dataset.csv")
texts = df['text'].tolist()
labels = df['label'].map({'ham': 0, 'spam': 1}).tolist()  # Convert to 0/1

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Custom Dataset
class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # Now works since labels are 0/1
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_texts, train_labels)
val_dataset = SpamDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")

# Save Model
model.save_pretrained("spam-bert-custom")
tokenizer.save_pretrained("spam-bert-custom")
print("✅ Model saved to 'spam-bert-custom'")