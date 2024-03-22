import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
from data.create import DatasetCreator
from customs.custom_model import CustomModel
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load the dataset
dataset_creator = DatasetCreator('./Datasets/dataset.csv')
id2label, label2id, labels = dataset_creator.label_maps
x_train, y_train, x_test, y_test = dataset_creator.get_splits()
x_train = x_train[:1000]
y_train = y_train[:1000]
x_train.append('Bootcut denim jeans')
y_train.append(label2id['bottoms'])
# Load the model and tokenizer
model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
base_model = AutoModel.from_pretrained(model_checkpoint)


model = CustomModel(base_model, labels)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, labels, inputs, mode=False):
        ce_loss = F.cross_entropy(predictions, labels)
        if not mode:
            return ce_loss

        penalty = 0
        inputs = inputs.lower()

        if predictions.argmax().item() == label2id['tops']:
            if ('shirt' not in inputs) or ('t-shirt' not in inputs):
                penalty *= 2
            elif ('basic' in inputs) or ('cotton' in inputs) or ('plain' in inputs) or ('polo' in inputs):
                penalty *= 0.75
        elif predictions.argmax().item() == label2id['bottoms']:
            if ('jeans' not in inputs):
                penalty += 0.5
            else:
                if ('straight' not in inputs) or ('regular' not in inputs) or ('bootcut' in inputs):
                    penalty *= 2
                else:
                    penalty *= 0.75
        penalty += len(inputs.split()) * 0.1

        total_loss = ce_loss + penalty

        return total_loss


criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=8e-4)


def train_epoch(model, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for i, _ in enumerate(x_train):
        inputs = x_train[i]
        label = torch.tensor(y_train[i]).to(device)

        inputs = tokenizer(inputs, padding=True,
                           truncation=True, return_tensors="pt").to(device)
        optimizer.zero_grad()
        outputs = model(**inputs).to(device)

        loss = criterion(outputs.flatten(), label, x_train[i], True)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, {i}/{len(x_train)}, Loss: {loss.item()}")
    print(f"Loss: {total_loss / len(x_train)}")


def evaluate(model):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for i in range(len(x_test)):
            inputs = x_train[i]
            label = torch.tensor(y_train[i])

            inputs = tokenizer(inputs, padding=True,
                               truncation=True, return_tensors="pt").to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask)

            loss = criterion(outputs, label, x_train[i])
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == label).sum().item()
            print(f"Loss:{loss}, Predict:{
                  id2label[predicted.item()]}, Actual:{id2label[label.item()]}")
    return total_correct / len(x_test)


# Training loop
num_epochs = 5
model.to(device)
for epoch in range(num_epochs):
    train_epoch(model, optimizer, criterion, epoch)

torch.save(model, './saved-models/distilbert-custom.pt')
tokenizer.save_pretrained('./saved-models/distilbert-custom-tokenizer')
