from datasets import Dataset
import pandas as pd

TRAIN_RATIO = 0.9


def create_dataset():
    df = pd.read_csv('Datasets/dataset.csv', encoding='latin1')
    hf_dataset = Dataset.from_pandas(df)

    hf_dataset = hf_dataset.train_test_split(
        test_size=1-TRAIN_RATIO, shuffle=True)
    return hf_dataset
