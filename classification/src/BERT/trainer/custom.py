import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
from data.create import DatasetCreator
from customs.custom_model import CustomModel
import torch.nn.functional as F

# Load the dataset
dataset_creator = DatasetCreator('datasets/dataset.csv')
id2label, label2id, labels = dataset_creator.label_maps
x_train, y_train, x_test, y_test = dataset_creator.train_test_split
x_train = x_train[:1090]
y_train = y_train[:1090]

# Load the model and tokenizer
model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
base_model = AutoModel.from_pretrained(model_checkpoint)


model = CustomModel(base_model, labels)


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, labels, inputs):
        ce_loss = F.cross_entropy(predictions, labels)

        penalty = 0

        if predictions.argmax() == label2id['tops']:
            if 'shirt' not in inputs:
                penalty += 0.5

        total_loss = ce_loss + penalty

        return total_loss


criterion = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)


def train_epoch(model, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for i, _ in enumerate(x_train):
        inputs = x_train[i]
        label = torch.tensor(y_train[i])

        inputs = tokenizer(inputs, padding=True,
                           truncation=True, return_tensors="pt")
        optimizer.zero_grad()
        outputs = model(**inputs)

        loss = criterion(outputs.flatten(), label, x_train[i])
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
                               truncation=True, return_tensors="pt")
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
num_epochs = 1
for epoch in range(num_epochs):
    train_epoch(model, optimizer, criterion, epoch)
torch.save(model, './saved-models/distilbert-custom.pt')

print(evaluate(model))
