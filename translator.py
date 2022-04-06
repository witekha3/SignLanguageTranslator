import config
from sign_trainer import SignTrainer
import numpy as np

class Translator:

    def __init__(self):
        self.trainer = SignTrainer()
        self.model = self.trainer.load_model()
        self.translations = list(self.trainer.label_map)

    def translate(self, sequence):
        sequence = self.trainer.pad_sequence(sequence)
        predictions =  self.model.predict(np.expand_dims(sequence, axis=0))[0]

        if predictions[np.argmax(predictions)] >= config.THRESHOLD:
            return self.translations[np.argmax(predictions)]
        else:
            return None
