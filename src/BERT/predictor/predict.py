from transformers import (
    AutoTokenizer)
import torch
import pandas as pd

TEST_DATA = 'Datasets/test_sample.csv'

df = pd.read_csv(TEST_DATA, encoding='latin1')
text_list = []
for index, row in df.iterrows():
    text_list.append((row['category'], row['name']))

MODEL_NAME = 'models/bert-lora.pt'
TOKENIZER_NAME = 'models/bert-lora-tokenizer'
id2label = {0: 'accessories', 1: 'beauty', 2: 'socks', 3: 'bottoms', 4: 'tops',
            5: 'outerwear', 6: 'shoes', 7: 'sportswear', 8: 'underwear', 9: 'swimwear'}
label2id = {'accessories': 0, 'beauty': 1, 'socks': 2, 'bottoms': 3, 'tops': 4,
            'outerwear': 5, 'shoes': 6, 'sportswear': 7, 'underwear': 8, 'swimwear': 9}

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

for truth, text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    logits = model(inputs).logits
    prediction_idxs = torch.max(logits, 1).indices
    prediction = id2label[prediction_idxs.tolist()[0]]
    predictions.append(prediction)
    if prediction == truth:
        correct += 1
    print(f'Predicted: {prediction}, for: {text}')
print(f'Accuracy: {correct/total}')


# df['predictions'] = predictions
# df.to_csv('predictions.csv', index=False)
