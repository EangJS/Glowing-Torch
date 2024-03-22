from data.create import DatasetCreator
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer

dataset_creator = DatasetCreator('datasets/custom_dataset.csv')
x_train, y_train, x_test, y_test = dataset_creator.get_splits()
id2label, label2id, labels = dataset_creator.label_maps
dataset = dataset_creator.dataset

# Load SetFit model from Hub
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-mpnet-base-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20,  # Number of text pairs to generate for contrastive learning
    num_epochs=1  # Number of epochs to use for contrastive learning
)

# Train and evaluate!
trainer.train()
metrics = trainer.evaluate()
print(metrics)
