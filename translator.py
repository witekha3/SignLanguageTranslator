from sign_trainer import SignTrainer


class Translator:

    def __init__(self):
        self.trainer = SignTrainer()
        self.model = self.trainer.load_model()

    def translate(self, sequence):
        sequence = self.trainer.pad_sequence(sequence)
        return self.model.predict(sequence)[0]
