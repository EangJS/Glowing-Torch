import json
import torch

device = 'cuda'
MODEL_NAME = './saved-models/bert-new-vanilia.pt'
model = torch.load(MODEL_NAME)
model.to(device)

with open('saved-models/label_maps.json', 'r') as f:
    label_maps = json.load(f)
    id2label = label_maps['id2label']
    label2id = label_maps['label2id']


text_list = ['Wireless Bra (Ultra Relax)', 'Cycling shorts', 'Short Sleeve Graphic Tee-Shirt', 'Standard Colour Jeans', 'Cotton Color Shirt', 'Kakhi Shorts', '3/4 Shorts',
             'Frill-trimmed denim blouse', 'Basquiat UT (Short Sleeve Graphic T-Shirt)', 'straight jeans', 'chelsea boots', 'standard bootcut jeans', 'bootcut jeans']

for i in text_list:
    pred = model([i]).item()
    print(f"{i} -> {id2label[str(pred)]}")


