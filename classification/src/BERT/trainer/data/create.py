from datasets import Dataset, DatasetDict
import pandas as pd
import json


class DatasetCreator:
    def __init__(self, file_path: str, train_ratio: float = 0.8):
        self.train_ratio = train_ratio
        self.file_path = file_path
        self.__hf_dataset = self.read_csv_dataset()
        self.label_maps = self.__make_label_maps()
        self.train_test_split = self.__map_labels()

    def read_csv_dataset(self):
        df = pd.read_csv(self.file_path, encoding='latin1')
        hf_dataset = Dataset.from_pandas(df)
        hf_dataset = hf_dataset.train_test_split(
            test_size=1-self.train_ratio, shuffle=True)
        return hf_dataset

    def __make_label_maps(self):
        y_train = self.__hf_dataset['train']['category']
        y_test = self.__hf_dataset['test']['category']
        labels = set()
        for label in y_test:
            labels.add(label)
        for label in y_train:
            labels.add(label)
        self.__id2label = {}
        self.__label2id = {}
        for i, label in enumerate(labels):
            self.__id2label[i] = label
            self.__label2id[label] = i
        label_dict = {"id2label": self.__id2label, "label2id": self.__label2id}
        with open('saved-models/label_maps.json', 'w') as f:
            json.dump(label_dict, f, indent=4)
        return self.__id2label, self.__label2id, labels

    def __map_labels(self):
        # shuffle the dataset just in case we want to use a subset
        train = self.__hf_dataset['train'].shuffle(seed=42)
        test = self.__hf_dataset['test'].shuffle(seed=42)
        y_train = train['category']
        y_test = test['category']
        for i, _ in enumerate(y_train):
            y_train[i] = self.__label2id[y_train[i]]
        for i, _ in enumerate(y_test):
            y_test[i] = self.__label2id[y_test[i]]
        x_test = self.__hf_dataset['test']['name']
        x_train = self.__hf_dataset['train']['name']
        self.dataset = DatasetDict({'train': Dataset.from_dict({'label': y_train, 'text': x_train}),
                                   'validation': Dataset.from_dict({'label': y_test, 'text': x_test})})

        return x_train, y_train, x_test, y_test
