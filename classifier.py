import torch
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("trained_classifier")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.eval()

LABELS = ["Invoice", "Resume", "Utility Bill", "Other", "Unclassifiable"]

def classify(text):

    if not text or text.strip() == "":
        return "Unclassifiable", 0.0
    
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=1)[0]
    label = torch.argmax(probs).item()

    return LABELS[label], float(probs[label])
