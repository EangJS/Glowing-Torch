from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification)
from datasets import DatasetDict, Dataset
from data.create import create_dataset

dataset = create_dataset()

x_train = dataset['train']['name']
y_train = dataset['train']['category']

x_test = dataset['test']['name']
y_test = dataset['test']['category']


def label_x_id():
    labels = set()
    for label in y_test:
        labels.add(label)
    for label in y_train:
        labels.add(label)
    id2label = {}
    label2id = {}
    for i, label in enumerate(labels):
        id2label[i] = label
        label2id[label] = i

    return id2label, label2id, labels


id2label, label2id, labels = label_x_id()

for i, _ in enumerate(y_train):
    y_train[i] = label2id[y_train[i]]
for i, _ in enumerate(y_test):
    y_test[i] = label2id[y_test[i]]

dataset = DatasetDict({'train': Dataset.from_dict({'label': y_train, 'text': x_train}),
                       'validation': Dataset.from_dict({'label': y_test, 'text': x_test})})


model_checkpoint = 'distilbert-base-uncased'

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id)

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, add_prefix_space=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def tokenize_function(examples):
    text = examples["text"]
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)

try:
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
except Exception:
    print("Continue")
tokenized_dataset.set_format("torch")


small_train_dataset = tokenized_dataset["train"].shuffle(
    seed=42) .select(range(1000))
small_eval_dataset = tokenized_dataset["validation"].shuffle(
    seed=42) .select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True)
eval_dataloader = DataLoader(small_eval_dataset)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)


text_list = ["Adidas tracking jersey", "Patterned dress", "Slim Fit Cotton twill trousers", "Relaxed Fit Denim jacket",
             "Stretch Fleece Mock Neck Long Sleeve T-Shirt", "Rudolph the Red-Nosed Reindeer Finger Puppets"]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)
    print(inputs)

    print(text + " - " + id2label[predictions.tolist()])


loss = None
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    print(f"Epoch:{epoch}, Loss :{loss}")

model.eval()
text_list = ["Patterned dress", "Slim Fit Cotton twill trousers", "Relaxed Fit Denim jacket",
             "Stretch Fleece Mock Neck Long Sleeve T-Shirt", "Rudolph the Red-Nosed Reindeer Finger Puppets", "Kids cute T-Shirt"]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to('cuda')
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(inputs)

    print(text + " - " + id2label[predictions.tolist()])


text_list = ["Adidas tracking jersey", "Patterned dress", "Slim Fit Cotton twill Down", "Prospex Land Meachanical Timepiece", "CoCo Crush Earring",
             "EAU DE Parfum Spray", "3-dimensional tweed by setting the entire piece with stones.", "Sardines", "Kids cute T-Shirt"]
print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(
        "cuda")
    logits = model(inputs).logits
    predictions = torch.max(logits, 1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])

torch.save(model, './models/bert-vanilia.pt')
tokenizer.save_pretrained('./models/bert-vanilia-tokenizer')
