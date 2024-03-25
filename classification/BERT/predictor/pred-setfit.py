import torch
import pandas as pd
import json

TEST_DATA = 'datasets/custom_dataset.csv'
MODEL_NAME = './saved-models/clothing-setfit.pt'

df = pd.read_csv(TEST_DATA, encoding='latin1')

with open('saved-models/label_maps.json', 'r') as f:
    label_maps = json.load(f)
    id2label = label_maps['id2label']
    label2id = label_maps['label2id']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load(MODEL_NAME) if device == 'cuda' else torch.load(MODEL_NAME, map_location=torch.device('cpu'))
model.to(device)

correct = 0
text_list = ['Wireless Bra', 'Short Sleeve Graphic Tee-Shirt', 'Standard Colour Jeans', 'Cotton Color Shirt', 'Kakhi Shorts', '3/4 Shorts',
             'Knitted Polo T-Shirt', 'Basquiat UT T-Shirt', 'straight jeans', 'Super Stretch Jeans', 'standard bootcut jeans']
total = len(text_list)
predictions = model.predict_proba(text_list)
for idx,i in enumerate(predictions):
    probs = torch.softmax(i, dim=0)
    max_index = torch.argmax(probs).item()
    print(f"{text_list[idx]} -> Prediction: {id2label[str(max_index)]} with confidence: {probs[max_index].item()}")

