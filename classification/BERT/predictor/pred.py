from transformers import (
    AutoTokenizer)
import torch
import pandas as pd
import json

TEST_DATA = 'Datasets/dataset.csv'
MODEL_NAME = './saved-models/bert-vanilia.pt'
TOKENIZER_NAME = './saved-models/distilbert-custom-tokenizer'

df = pd.read_csv(TEST_DATA, encoding='latin1')

with open('saved-models/label_maps.json', 'r') as f:
    label_maps = json.load(f)
    id2label = label_maps['id2label']
    label2id = label_maps['label2id']

device = 'cpu'  # default device
if torch.cuda.is_available():
    model = torch.load(MODEL_NAME)
    device = 'cuda'
else:
    model = torch.load(MODEL_NAME, map_location=torch.device(device))

model.to(device)

classifier_params = model.classifier.parameters()
for param in classifier_params:
    print(param)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME, add_prefix_space=True)

predictions = []
correct = 0
text_list = ['Wireless Bra (Ultra Relax)', 'Cycling shorts', 'Short Sleeve Graphic Tee-Shirt', 'Standard Colour Jeans', 'Cotton Color Shirt', 'Kakhi Shorts', '3/4 Shorts',
             'Frill-trimmed denim blouse', 'Basquiat UT (Short Sleeve Graphic T-Shirt)', 'straight jeans', 'chelsea boots', 'standard bootcut jeans']
total = len(text_list)


for text in text_list:
    inputs = tokenizer(text, padding=True,
                       truncation=True, return_tensors="pt").to(device)

    output = model(**inputs)
    predictions = torch.max(output.logits, 1).indices
    prediction: str = id2label[str(predictions.tolist()[0])]
    print(f"Predicted: {prediction}, for: {text}")
