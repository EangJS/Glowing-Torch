from transformers import (
    AutoTokenizer)
import torch
import pandas as pd
import evaluator

TEST_DATA = 'Datasets/dataset.csv'
MODEL_NAME = './models/distilbert-lora.pt'
TOKENIZER_NAME = './models/distilbert-lora-tokenizer'

df = pd.read_csv(TEST_DATA, encoding='latin1')
text_list = []
for index, row in df.iterrows():
    text_list.append((row['category'], row['name']))

id2label = {0: 'accessories', 1: 'sportswear', 2: 'bottoms', 3: 'tops',
            4: 'socks', 5: 'outerwear', 6: 'underwear', 7: 'shoes', 8: 'swimwear'}
label2id = {'accessories': 0, 'sportswear': 1, 'bottoms': 2, 'tops': 3,
            'socks': 4, 'outerwear': 5, 'underwear': 6, 'shoes': 7, 'swimwear': 8}

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
total = len(text_list)

x_test = [x[1] for x in text_list]
y_test = [x[0] for x in text_list]

predictions = evaluator.test(
    model, tokenizer, x_test, y_test, id2label, label2id, device)

evaluator.evaluate_f1(y_test, predictions)


# df['predictions'] = predictions
# df.to_csv('predictions.csv', index=False)
