import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
from data.create import DatasetCreator


dataset_creator = DatasetCreator()
id2label, label2id, labels = dataset_creator.get_label_maps()
x_train, y_train, x_test, y_test = dataset_creator.get_train_test_split()
model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
base_model = AutoModel.from_pretrained(model_checkpoint)


class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(768, len(labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(
            # Use the output of the [CLS] token
            outputs.last_hidden_state[:, 0, :])
        return logits


model = CustomModel(base_model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)


def train_epoch(model, optimizer, criterion):
    model.train()
    total_correct = 0
    for i in range(len(x_train)):
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
        print(f"Predict:{predicted.item()},Actual:{label}")
        loss.backward()
        print(loss.item())
        optimizer.step()


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
    train_acc = train_epoch(
        model, optimizer, criterion)

print(evaluate(model))
