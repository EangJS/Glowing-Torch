from datasets import DatasetDict, Dataset
from data.create import create_dataset
import numpy as np
import torch
import evaluate
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

model_checkpoint = 'distilbert-base-uncased'
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, add_prefix_space=True)


def get_dataset():
    dataset = create_dataset()

    x_train = dataset['train']['name'][:1000]
    y_train = dataset['train']['category'][:1000]

    x_test = dataset['test']['name']
    y_test = dataset['test']['category']
    return x_train, y_train, x_test, y_test


def label_x_id(y_train, y_test):
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


def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}


def train(tokenized_dataset, data_collator, model, tokenizer, compute_metrics):
    peft_config = LoraConfig(task_type="SEQ_CLS",
                             r=4,
                             lora_alpha=32,
                             lora_dropout=0.01,
                             target_modules=['q_lin'])

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    lr = 1e-3
    batch_size = 4
    num_epochs = 1

    training_args = TrainingArguments(
        output_dir=f"models/bert-lora-text-classification",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    torch.save(trainer.model, './models/distilbert-lora.pt')


def test(model, tokenizer, x_test, y_test, id2label):
    correct = 0
    for text in x_test:
        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices
        prediction = id2label[predictions.tolist()[0]]
        actual = id2label[y_test[x_test.index(text)]]
        if prediction == actual:
            correct += 1
            print(f"Predicted Correctly: {prediction}, for: {text}")
        else:
            print(f"Predicted Wrongly: {
                  prediction} - Actual: {actual}, for: {text}")
    accuracy = correct / len(x_test)
    print(f'Accuracy: {accuracy}')


def main():
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    x_train, y_train, x_test, y_test = get_dataset()
    id2label, label2id, labels = label_x_id(y_train, y_test)
    for i, _ in enumerate(y_train):
        y_train[i] = label2id[y_train[i]]
    for i, _ in enumerate(y_test):
        y_test[i] = label2id[y_test[i]]
    dataset = DatasetDict({'train': Dataset.from_dict({'label': y_train, 'text': x_train}),
                           'validation': Dataset.from_dict({'label': y_test, 'text': x_test})})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id)
    model.to(device)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train(tokenized_dataset, data_collator, model, tokenizer, compute_metrics)
    test(model, tokenizer, x_test, y_test, id2label)

    tokenizer.save_pretrained('./models/distilbert-lora-tokenizer')


if __name__ == "__main__":
    main()
