from datasets import Dataset, DatasetDict
import pandas as pd
import json
TRAIN_RATIO = 0.98


def create_dataset():
    df = pd.read_csv('Datasets/dataset.csv', encoding='latin1')
    hf_dataset = Dataset.from_pandas(df)

    hf_dataset = hf_dataset.train_test_split(
        test_size=1-TRAIN_RATIO, shuffle=True, seed=42)
    return hf_dataset


class DatasetCreator:
    def __init__(self, file_path: str, train_ratio: float = 0.9):
        self.train_ratio = train_ratio
        self.file_path = file_path
        self.__hf_dataset = self.read_csv_dataset()
        self.__get_raw_splits()
        self.label_maps = self.__label_x_id()
        self.__set_label_index()
        self.dataset = DatasetDict({'train': Dataset.from_dict({'label': self.y_train, 'text': self.x_train}),
                                    'validation': Dataset.from_dict({'label': self.y_test, 'text': self.x_test})})

    def read_csv_dataset(self):
        df = pd.read_csv(self.file_path, encoding='latin1')
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset = hf_dataset.train_test_split(
            test_size=1-self.train_ratio, shuffle=True, seed=42)
        return hf_dataset

    def __get_raw_splits(self):
        self.x_train = self.__hf_dataset['train']['name']
        self.y_train = self.__hf_dataset['train']['category']
        self.x_test = self.__hf_dataset['test']['name']
        self.y_test = self.__hf_dataset['test']['category']

    def __label_x_id(self):
        labels = set()
        for label in self.y_test:
            labels.add(label)
        for label in self.y_train:
            labels.add(label)
        id2label = {}
        label2id = {}
        for i, label in enumerate(labels):
            id2label[i] = label
            label2id[label] = i
        self.label2id = label2id
        with open('saved-models/label_maps.json', 'w') as f:
            json.dump({'id2label': id2label, 'label2id': label2id}, f, indent=4)

        return id2label, label2id, labels

    def __set_label_index(self):
        for i, _ in enumerate(self.y_train):
            self.y_train[i] = self.label2id[self.y_train[i]]
        for i, _ in enumerate(self.y_test):
            self.y_test[i] = self.label2id[self.y_test[i]]

    def get_splits(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
