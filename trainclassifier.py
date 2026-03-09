import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from pdfminer.high_level import extract_text

LABELS = {
    "invoice": 0,
    "resume": 1,
    "utilitybill": 2,
    "other": 3,
    "unclassifiable": 4
}

class PDFDataset(Dataset):
    def __init__(self, folder):
        self.samples = []

        for file in os.listdir(folder):
            if not file.endswith(".pdf"):
                continue

            label_key = file.split("_")[0]
            label = LABELS.get(label_key, 4)

            try:
                text = extract_text(os.path.join(folder, file))
            except Exception:
                print(f"Skipping bad PDF: {file}")
                continue

            text = text[:512]
            self.samples.append((text, label))

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label)

        return item


dataset = PDFDataset("documents")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5
)

optimizer = AdamW(model.parameters(), lr=2e-5)

model.train()

for epoch in range(5):
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss:", loss.item())

model.save_pretrained("trained_classifier")
print("Training complete. Model saved.")
