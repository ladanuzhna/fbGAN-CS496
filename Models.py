import tensorflow as tf

class Generator(tf.keras.Model):

    def __init__(self,input_size = 50):
        super().__init__(name='generator')

    def optimizer(self):
        pass

    def train(self):
        pass

    def objective(self):
        pass

    def test(self):
        pass


class Discriminator(tf.keras.Model):

    def __init__(self):
        super().__init__(name='discriminator')

    def optimizer(self):
        pass

    def train(self):
        pass

    def objective(self):
        pass

    def test(self):
        pass

class Feedback(tf.keras.Model):

    def __init__(self):
        pass

    def optimizer(self):
        pass

    def objective(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

