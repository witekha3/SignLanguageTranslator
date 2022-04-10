import os
import shutil

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Masking, LSTM, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split

import config
from actions_collector.collect_data import load_smaller_dataset
from body_detecotr import BodyDetector
from body_detecotr import POINTS_NUM


class SignTrainer:

    def __init__(self):
        # self.data = BodyDetector.get_points()
        #only for test
        self.data = load_smaller_dataset()
        ###
        self.label_map = {label: num for num, label in enumerate(set(self.data.index.to_list()))}
        self.max_sequence_len = self.data[list(POINTS_NUM.keys())[0]].map(len).max()
        self.min_sequence_len = self.data[list(POINTS_NUM.keys())[0]].map(len).min()
        self.special_val = -10
        self.model_path = os.path.join(config.TENSOR_DIR, "model.h5")

    @staticmethod
    def _prepare_dir_for_logs():
        shutil.rmtree(config.TENSOR_DIR, ignore_errors=True)
        os.makedirs(config.TENSOR_DIR)

    def _prepare_model(self):
        model = Sequential()
        # act !!relu!! or ?tanh?
        model.add(Masking(mask_value=self.special_val, input_shape=(self.max_sequence_len, sum(POINTS_NUM.values()))))
        model.add(LSTM(64, return_sequences=True, activation='tanh'))
        model.add(LSTM(128, return_sequences=True, activation='tanh'))
        model.add(LSTM(64, return_sequences=False, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(len(self.label_map), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def pad_sequence(self, sequence):
        return np.pad(sequence, [(self.max_sequence_len - len(sequence), 0), (0, 0)], 'constant',
                      constant_values=self.special_val)

    def train_generator(self):
        features = []
        train = []
        for index, row in self.data.iterrows():
            # Padding to max sequence len
            x_train = self.pad_sequence(row["data"])
            y_train = self.label_map[row["action"]]

            features.append(x_train)
            train.append(y_train)
        return train_test_split(np.array(features), to_categorical(train).astype(int), test_size=0.3)

    def train_data(self, save=False):
        if save:
            self._prepare_dir_for_logs()

        tensor_board = TensorBoard(log_dir=config.TENSOR_DIR)
        # patient early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1,
                           patience=200)  # Available metrics are: loss,categorical_accuracy
        model = self._prepare_model()
        x_train, x_test, y_train, y_test = self.train_generator()
        model.fit(x_train, y_train, epochs=config.EPOCHS, callbacks=[tensor_board, es],
                  validation_data=(x_test, y_test))

        # TODO: https://datascience.stackexchange.com/questions/48796/how-to-feed-lstm-with-different-input-array-sizes
        # model.fit(self.train_generator(), epochs=config.EPOCHS, callbacks=[tensor_board, es])
        model.summary()
        if save:
            model.save(self.model_path)
        return model

    def load_model(self):
        model = self._prepare_model()
        model.load_weights(self.model_path)
        return model


# a = SignTrainer()
# a.train_data(False) #NOT SAVING
