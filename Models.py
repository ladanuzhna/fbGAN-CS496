import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Input,Dense

SEQ_LENGTH = 128
HIDDEN_UNITS = 100
KERNEL_SIZE = 5
BATCH_SIZE = 1

class ResidualBlock(tf.keras.Model):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.model = tf.keras.models.Sequential([
                        Input(shape=(BATCH_SIZE,HIDDEN_UNITS)),
                        Dense(HIDDEN_UNITS,activation='relu'),
                        Conv1D(filters=HIDDEN_UNITS,kernel_size=KERNEL_SIZE,padding='same',strides=1,activation='relu'),
                        Conv1D(filters=HIDDEN_UNITS,kernel_size=KERNEL_SIZE,padding='same',strides=1)])

    def __call__(self,X,alpha = 0.3):
        return X + alpha*self.model(X)

class Generator(tf.keras.Model):

    def __init__(self):
        """
        implementation of Generator
        :param input_size: size of the sequence (input noise)
        """
        super().__init__(name='generator')
        self.input = Dense(SEQ_LENGTH,SEQ_LENGTH*HIDDEN_UNITS)
        self.residual = tf.keras.models.Sequential([ResidualBlock(),ResidualBlock(),ResidualBlock(),
                                                    ResidualBlock(),ResidualBlock()])

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

"""
We can try different architectures for Feedback Analyzer here!
I would follow the convention of creating Feedback_1 ... Feedback_2
Can also compare the performance of these 
"""

class Feedback_1(tf.keras.Model):

    def __init__(self):
        pass

    def __call__(self,X,training):
        """

        :param X: input of the size [batch_size, seq_length];
        :param training: specifies the behavior of the call;
        :return: Y: scores for the input sequences;
        """
        pass

