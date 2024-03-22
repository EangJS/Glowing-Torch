from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification)
from data.create import DatasetCreator
from predictor.evaluator import test

dataset_creator = DatasetCreator('./Datasets/custom_dataset.csv')
id2label, label2id, labels = dataset_creator.label_maps
x_train, y_train, x_test, y_test = dataset_creator.get_splits()
dataset = dataset_creator.dataset

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
    seed=42)  # .select(range(1000))
small_eval_dataset = tokenized_dataset["validation"].shuffle(
    seed=42)  # .select(range(1000))

train_dataloader = DataLoader(small_train_dataset, shuffle=True)
eval_dataloader = DataLoader(small_eval_dataset)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)


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

torch.save(model, './saved-models/bert-vanilia.pt')
tokenizer.save_pretrained('./saved-models/bert-vanilia-tokenizer')

test(model, tokenizer, x_test, y_test, id2label, label2id, device)
