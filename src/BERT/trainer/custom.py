import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
from data.create import DatasetCreator
from custom_model import CustomModel

# Load the dataset
dataset_creator = DatasetCreator('Datasets/dataset.csv')
id2label, label2id, labels = dataset_creator.label_maps
x_train, y_train, x_test, y_test = dataset_creator.train_test_split

# Load the model and tokenizer
model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
base_model = AutoModel.from_pretrained(model_checkpoint)


model = CustomModel(base_model, labels)

criterion = nn.CrossEntropyLoss()
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

        loss = criterion(outputs.flatten(), label)
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

            loss = criterion(outputs.flatten(), label)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == label).sum().item()
            print(f"Loss:{loss}, Predict:{
                  id2label[predicted.item()]}, Actual:{id2label[label.item()]}")
    return total_correct / len(x_test)


# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    train_epoch(model, optimizer, criterion, epoch)
torch.save(model, './models/distilbert-custom.pt')

print(evaluate(model))
