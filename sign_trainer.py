import glob
import os
from os import listdir
from os.path import isfile, join
import shutil
import numpy as np
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.layers import Masking, LSTM, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config
from body_detector import BodyDetector, ACTIONS_DIR
from body_detector import POINTS_NUM


class SignTrainer:

    def __init__(self):
        # for index, row in self.data.iterrows():
        #     for
        self.label_map = {label: num for num, label in enumerate(list(BodyDetector.get_all_actions_names()))}
        # self.max_sequence_len = self.data[list(POINTS_NUM.keys())[0]].map(len).max()
        self.max_sequence_len = 50
        self.min_sequence_len = 50
        # self.min_sequence_len = self.data[list(POINTS_NUM.keys())[0]].map(len).min()
        self.ignor_val = -10
        self.model_path = os.path.join(config.TENSOR_DIR, "model.h5")

    @staticmethod
    def _prepare_dir_for_logs():
        shutil.rmtree(config.TENSOR_DIR, ignore_errors=True)
        os.makedirs(config.TENSOR_DIR)

    def _prepare_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, activation='tanh', input_shape=(self.max_sequence_len, sum(POINTS_NUM.values())))),
        model.add(LSTM(80, return_sequences=True, activation='tanh'))
        model.add(LSTM(150, return_sequences=False, activation='tanh'))
        model.add(Dense(100, activation='relu')),
        model.add(Dense(50, activation='relu')),
        model.add(Dense(len(self.label_map), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def pad_sequence(self, sequence):
        return np.pad(sequence, [(0, self.max_sequence_len - len(sequence)), (0, 0)], 'constant',
                      constant_values=self.ignor_val)

    def train_generator(self):
        features = []
        train = []
        all_actions_names = BodyDetector.get_all_actions_names()
        for action_name in all_actions_names:
            last_repeat = BodyDetector.find_last_action_repeat(action_name)
            for i in range(last_repeat):
                x_train = BodyDetector.flatten_action(BodyDetector.get_points(action_name, i).iloc[0])
                y_train = self.label_map[action_name]
                features.append(np.array(x_train))
                train.append(y_train)
                # yield x_train, y_train
        return train_test_split(np.array(features), to_categorical(train).astype(int), test_size=0.2)

    def train_data(self, save=False):
        if save:
            self._prepare_dir_for_logs()
        model = self._prepare_model()
        x_train, x_test, y_train, y_test = self.train_generator()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch',
        )
        if save:
            tensor_board = TensorBoard(log_dir=config.TENSOR_DIR)
            es = EarlyStopping(monitor="val_loss", verbose=1, patience=10000)
            model.fit(x_train, y_train, epochs=config.EPOCHS, callbacks=[tensor_board, es, cp_callback],
                      validation_data=(x_test, y_test))
        else:
            model.fit(x_train, y_train, epochs=config.EPOCHS, validation_data=(x_test, y_test))
        model.summary()
        if save:
            model.save(self.model_path)
        return model

    def load_model(self):
        model = self._prepare_model()
        model.load_weights(self.model_path)
        return model

#
# a = SignTrainer()
# a.train_data(True)
