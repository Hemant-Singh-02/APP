from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer once
tokenizer = BertTokenizer.from_pretrained('spam-bert-custom')
model = BertForSequenceClassification.from_pretrained('spam-bert-custom')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# FastAPI setup
app = FastAPI(title="Spam Detection API")

# Request body schema
class TextRequest(BaseModel):
    text: str

# Prediction function
def model_predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return 'SPAM' if prediction == 1 else 'HAM'

# POST endpoint
@app.post("/predict")
def predict_spam(request: TextRequest):
    result = model_predict(request.text)
    return {"prediction": result}