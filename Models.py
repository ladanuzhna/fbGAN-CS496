import tensorflow as tf

SEQ_LENGTH = 128
HIDDEN_UNITS = 100
PADDING = 2
KERNEL_SIZE = 5

class ResidualBlock(tf.keras.Model):

    def __init__(self):
        self.relu = tf.nn.relu(HIDDEN_UNITS)
        self.conv1d = tf.nn.conv1d(input=HIDDEN_UNITS,padding=PADDING)

    def __call__(self):
        pass

class Generator(tf.keras.Model):

    def __init__(self,input_size = SEQ_LENGTH):
        """
        implementation of Generator
        :param input_size: size of the sequence (input noise)
        """
        super().__init__(name='generator')

    def __call__(self, X, training = False):
        """
        model's forward pass
        :param X: input of the size [batch_size, seq_length];
        :param training: specifies the behavior of the call
        :return: Z - batch of generated sequences
        """


class Discriminator(tf.keras.Model):

    def __init__(self, clip = 1):
        """
        implementation of Discriminator
        :param clip: value to which you clip the gradients (or False)
        """
        super().__init__(name='discriminator')

    def call(self,X,training = False):
        """
        model's forward pass
        :param X: input of the size [batch_size, seq_length];
        :param training: specifies the behavior of the call;
        :return: Y: probability of each sequences being real of shape [batch_size, 1]
        """
        pass

class Feedback(tf.keras.Model):

    def __init__(self):
        pass

    def __call__(self,X,training):
        """

        :param X: input of the size [batch_size, seq_length];
        :param training: specifies the behavior of the call;
        :return: Y: scores for the input sequences;
        """
        pass

