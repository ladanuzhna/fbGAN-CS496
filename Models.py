import tensorflow as tf

class Generator(tf.keras.Model):

    def __init__(self,input_size = 50):
        """
        implementation of Generator
        :param input_size: size of the sequence (input noise)
        """
        super().__init__(name='generator')

    def __call__(self, X):
        """
        model's forward pass
        :param X - input of the size [batch_size, seq_length]
        :return: Z - batch of generated sequences
        """


class Discriminator(tf.keras.Model):

    def __init__(self, clip = 1):
        """
        implementation of Discriminator
        :param clip: value to which you clip the gradients (or False)
        """
        super().__init__(name='discriminator')

    def call(self,X):
        """
        model's forward pass
        :param X - input of the size [batch_size, seq_length]
        :return: Y - probability of each sequences being real of shape [batch_size, 1]
        """
        pass

class Feedback(tf.keras.Model):

    def __init__(self):
        pass

    def __call__(self):
        pass

