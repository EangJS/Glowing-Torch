from datasets import Dataset
import pandas as pd
from datasets import DatasetDict, Dataset

TRAIN_RATIO = 0.9


class DatasetCreator:
    def __init__(self):
        self.__hf_dataset = self.read_csv_dataset()
        self.id2label, self.label2id, self.labels = self.__make_label_maps()
        self.x_train, self.y_train, self.x_test, self.y_test = self.__map_labels()
        self.dataset = DatasetDict({'train': Dataset.from_dict({'label': self.y_train, 'text': self.x_train}),
                                    'validation': Dataset.from_dict({'label': self.y_test, 'text': self.x_test})})

    def read_csv_dataset(self):
        df = pd.read_csv('Datasets/dataset.csv', encoding='latin1')
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset = hf_dataset.train_test_split(
            test_size=1-TRAIN_RATIO, shuffle=True)
        return hf_dataset

    def __make_label_maps(self):
        y_train = self.__hf_dataset['train']['category']
        y_test = self.__hf_dataset['test']['category']
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

    def __map_labels(self):
        y_train = self.__hf_dataset['train']['category']
        y_test = self.__hf_dataset['test']['category']
        for i, _ in enumerate(y_train):
            y_train[i] = self.label2id[y_train[i]]
        for i, _ in enumerate(y_test):
            y_test[i] = self.label2id[y_test[i]]
        x_test = self.__hf_dataset['test']['name']
        x_train = self.__hf_dataset['train']['name']
        return x_train, y_train, x_test, y_test

    def get_train_test_split(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_label_maps(self):
        return self.id2label, self.label2id, self.labels
