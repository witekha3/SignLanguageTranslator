import os
import shutil

from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping

import config
from body_detecotr import BodyDetector
from body_detecotr import POINTS_NUM


class SignTrainer:

    def __init__(self):
        self.data = BodyDetector.get_points()
        self.label_map = {label: num for num, label in enumerate(self.data["action"].unique())}
        self._prepare_dir_for_logs()
        self.model_path = os.path.join(config.TENSOR_DIR, "model.h5")

    @staticmethod
    def _prepare_dir_for_logs():
        shutil.rmtree(config.TENSOR_DIR, ignore_errors=True)
        os.makedirs(config.TENSOR_DIR)

    def _prepare_model(self):
        model = Sequential()
        # shape = (config.NUM_OF_FRAMES, sum([i for i in BPExtractor.POINTS_NUM.values()]))
        # act relu
        model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(None, sum(POINTS_NUM.values()))))
        model.add(LSTM(128, return_sequences=True, activation='tanh'))
        model.add(LSTM(64, return_sequences=False, activation='tanh'))
        model.add(Dense(64, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(len(self.label_map), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def train_generator(self):
        for index, row in self.data.iterrows():
            x_train = row["data"]
            y_train = to_categorical(self.label_map[row["action"]])
            yield x_train, y_train

    def train_data(self, save=False):
        tensor_board = TensorBoard(log_dir=config.TENSOR_DIR)
        # patient early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=200) #Available metrics are: loss,categorical_accuracy
        model = self._prepare_model()
        model.fit(self.train_generator(), epochs=config.EPOCHS, callbacks=[tensor_board, es])
        model.summary()
        if save:
            model.save(self.model_path)
        return model

    def load_model(self):
        model = self._prepare_model()
        model.load_weights(self.model_path)
        return model


a = SignTrainer()
a.train_data()
