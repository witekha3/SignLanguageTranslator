from sequence_translator import SequenceRecognizer, LiveVideoTranslator
from sign_trainer import SignTrainer
from translator import Translator

if __name__ == '__main__':
    # SignTrainer().train_model() # Training the model
    # Translator.english_to_asl("hello") # Show "Hello" in ASL
    # SequenceRecognizer("hello.mp4").start() # Translating the gesture from the video
    LiveVideoTranslator().start()  # Runs real-time gesture detection
