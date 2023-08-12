from tensorflow.keras.preprocessing.text import one_hot


class OneHotEncoder:
    def __init__(self):
        self.encoder = one_hot

    def encode(self,s,vocab_size):
        return self.encoder(s,vocab_size)