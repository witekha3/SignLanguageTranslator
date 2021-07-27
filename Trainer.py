import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

import config
from Actions import Actions
from BPExtractor import BPExtractor


class Trainer:

    def __init__(self):
        self.label_map = {label: num for num, label in enumerate(Actions.available_actions)}

    def fit_models(self):
        pass

    def save_models(self, models):
        pass

    def _prepare_dir_for_logs(self):
        shutil.rmtree(config.LOG_DIR, ignore_errors=True)
        os.makedirs(config.LOG_DIR)

    def _get_train_and_test_data(self):
        features, train = [], []
        for action in Actions.available_actions:
            all_action_repeats = np.split(BPExtractor.load(action), config.NUM_OF_CAPT_REPEATS)
            for repeat in all_action_repeats:
                features.append(repeat)
                train.append(self.label_map[action])
        return train_test_split(np.array(features), to_categorical(train).astype(int), test_size=config.TEST_SIZE)

    @staticmethod
    def _prepare_model():
        model = Sequential()
        shape = (config.NUM_OF_FRAMES, sum([i for i in BPExtractor.POINTS_NUM.values()]))
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=shape))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(Actions.available_actions.shape[0], activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def train_and_save_model(self):
        X_train, X_test, Y_train, Y_test = self._get_train_and_test_data()
        self._prepare_dir_for_logs()
        tensor_board = TensorBoard(log_dir=config.LOG_DIR)
        model = Trainer._prepare_model()
        model.fit(X_train, Y_train, epochs=config.EPOCHS, callbacks=[tensor_board])
        model.summary()
        model.save(config.MODEL_FILENAME)
        return model

    @staticmethod
    def load_model():
        model = Trainer._prepare_model()
        model.load_weights(config.MODEL_FILENAME)
        return model


    @staticmethod
    def predict(all_body_points, model):
        pass


