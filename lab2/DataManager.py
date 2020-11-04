import pickle
import numpy as np

from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, train_data_path, like_images=False):
        with open(train_data_path, 'rb') as fo:
            raw_train_data = pickle.load(fo, encoding='bytes')

        keys = list(raw_train_data.keys())
        self.train_labels = np.array(raw_train_data[keys[1]])
        self.train_data = np.array(raw_train_data[keys[2]])

        self.train_data = (self.train_data - np.mean(self.train_data, axis=0)) / np.std(self.train_data, axis=0)
        if like_images:
            self.train_data = self.train_data.reshape((len(self.train_data), 3, 32, 32))

        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(self.train_data,
                                                                                              self.train_labels,
                                                                                              test_size=0.1,
                                                                                              random_state=47)

    def get_test_batch(self, batch_size):
        return self.train_data[:batch_size], self.train_labels[:batch_size]

    def get_train_data(self):
        return self.train_data, self.val_data, self.train_labels, self.val_labels

