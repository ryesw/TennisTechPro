from tensorflow.keras.models import load_model


class ActionRecognition:
    def __init__(self):
        self.model = load_model('../models/model1.h5')