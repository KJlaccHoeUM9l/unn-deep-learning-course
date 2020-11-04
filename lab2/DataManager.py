import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self, dataset_root_path, like_images=False):
        def get_data_batch(data_path):
            with open(data_path, 'rb') as fo:
                raw_train_data = pickle.load(fo, encoding='bytes')
            keys = list(raw_train_data.keys())
            return np.array(raw_train_data[keys[2]]), np.array(raw_train_data[keys[1]])

        train_data_1, train_labels_1 = get_data_batch(os.path.join(dataset_root_path, 'data_batch_1'))
        train_data_2, train_labels_2 = get_data_batch(os.path.join(dataset_root_path, 'data_batch_2'))
        train_data_3, train_labels_3 = get_data_batch(os.path.join(dataset_root_path, 'data_batch_3'))
        train_data_4, train_labels_4 = get_data_batch(os.path.join(dataset_root_path, 'data_batch_4'))
        train_data_5, train_labels_5 = get_data_batch(os.path.join(dataset_root_path, 'data_batch_5'))
        self.test_data, self.test_labels = get_data_batch(os.path.join(dataset_root_path, 'test_batch'))

        self.train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5), axis=0)
        self.train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5), axis=0)

        self.train_data = (self.train_data - np.mean(self.train_data, axis=0)) / np.std(self.train_data, axis=0)
        self.test_data = (self.test_data - np.mean(self.test_data, axis=0)) / np.std(self.test_data, axis=0)
        if like_images:
            self.train_data = self.train_data.reshape((len(self.train_data), 3, 32, 32))
            self.test_data = self.test_data.reshape((len(self.test_data), 3, 32, 32))

        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(self.train_data,
                                                                                              self.train_labels,
                                                                                              test_size=0.1,
                                                                                              random_state=47)

    def get_test_batch(self, batch_size):
        return self.train_data[:batch_size], self.train_labels[:batch_size]

    def get_train_data(self):
        return self.train_data, self.val_data, self.train_labels, self.val_labels

    def get_test_data(self):
        return self.test_data, self.test_labels
