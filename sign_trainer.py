import json
import os
import shutil
from typing import List, Any

import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Masking, LSTM, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import tensorflow as tf

import config
from body_detector import BodyDetector
from body_detector import POINTS_NUM


class SignTrainer:
    """
    class responsible for gesture recognition network
    """

    def __init__(self):
        self.label_map = {label: num for num, label in enumerate(list(BodyDetector.get_all_actions_names()))}
        self.max_sequence_len = None
        self.min_sequence_len = None
        self._ignore_value = -10
        self._model_path = os.path.join(config.TENSOR_DIR, "model.h5")
        self._model_params = os.path.join(config.TENSOR_DIR, "model_params.json")

    @staticmethod
    def _prepare_dir_for_logs() -> None:
        """Prepare dir for tensorflow files like _model or tensorboard"""
        shutil.rmtree(config.TENSOR_DIR, ignore_errors=True)
        os.makedirs(config.TENSOR_DIR)

    def _prepare_model(self) -> Sequential:
        """Creates _model"""
        model = Sequential()
        model.add(Masking(mask_value=self._ignore_value, input_shape=(self.max_sequence_len, sum(POINTS_NUM.values()))))
        model.add(LSTM(250, return_sequences=True, activation='tanh')),
        model.add(LSTM(120, return_sequences=False, activation='tanh'))
        model.add(Dense(32, activation='relu')),
        model.add(Dense(len(self.label_map), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Padding sequence to max sequence len
        Example:
            - Max sequence len = 60
            - Given sequence len = 48
            The method will add 12 fragments filled with the value 'self._ignore_value' to the given sequence
        :param sequence: action sequence
        :return:
        """
        return np.pad(sequence, [(0, self.max_sequence_len - len(sequence)), (0, 0)], 'constant',
                      constant_values=self._ignore_value)

    def find_min_max_sequence_len(self, sequences_len: list) -> None:
        """
        Sets min and max sequence len
        :param sequences_len: list of sequences lengths
        :return: None
        """
        self.min_sequence_len = min(sequences_len)
        self.max_sequence_len = max(sequences_len)

    def test_train_generator(self) -> List[Any]:
        """
        Creates test and train data
        :return: x_train, x_test, y_train, y_test
        """
        features_tmp, features, train, sequences_len = [], [], [], []
        for action_name in BodyDetector.get_all_actions_names():
            last_repeat = BodyDetector.find_last_action_repeat(action_name)
            for i in range(last_repeat):
                x_train = BodyDetector.flatten_action(BodyDetector.get_points(action_name, i).iloc[0])
                y_train = self.label_map[action_name]
                sequences_len.append(x_train.shape[0])
                features_tmp.append(np.array(x_train))
                train.append(y_train)
        self.find_min_max_sequence_len(sequences_len)
        features = [self.pad_sequence(feature) for feature in features_tmp]
        return train_test_split(np.array(features), to_categorical(train).astype(int), test_size=0.2)

    def train_model(self) -> Sequential:
        """
        Trains _model
        :return: Model after training
        """
        self._prepare_dir_for_logs()
        x_train, x_test, y_train, y_test = self.test_train_generator()
        model = self._prepare_model()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self._model_path, verbose=1,)
        tensor_board = TensorBoard(log_dir=config.TENSOR_DIR)
        es = EarlyStopping(monitor="val_loss", verbose=1, patience=50)
        model.fit(x_train, y_train, epochs=config.EPOCHS, callbacks=[tensor_board, es, cp_callback], validation_data=(x_test, y_test))
        model.summary()
        model.save(self._model_path)
        with open(self._model_params, 'w') as f:
            json.dump({"min_seq_len": self.min_sequence_len, "max_seq_len": self.max_sequence_len}, f)
        return model

    def load_model(self) -> Sequential:
        """Loads saved _model"""
        model = self._prepare_model()
        model.load_weights(self._model_path)
        with open(self._model_params) as json_file:
            data = json.load(json_file)
            self.max_sequence_len = data["max_seq_len"]
            self.min_sequence_len = data["min_seq_len"]
        return model

