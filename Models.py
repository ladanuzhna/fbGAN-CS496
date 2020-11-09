import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Input,Dense, Reshape

SEQ_LENGTH = 100
HIDDEN_UNITS = 100
KERNEL_SIZE = 5
BATCH_SIZE = 1
N_CHAR = 20
N_SAMPLES = 150


def softmax(logits):
    shape = tf.shape(logits)
    res = tf.nn.softmax(tf.reshape(logits, [-1, N_CHAR])
    return tf.reshape(res, shape)


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.dense = Dense(HIDDEN_UNITS, activation='relu')
        self.conv1d_1 = Conv1D(filters=HIDDEN_UNITS, kernel_size=KERNEL_SIZE, padding='same', strides=1,
                               activation='relu')
        self.conv1d_2 = Conv1D(filters=HIDDEN_UNITS, kernel_size=KERNEL_SIZE, padding='same', strides=1)

    def __call__(self, X, alpha=0.3):
        x = self.dense(X)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        return x + alpha * self.model(x)


class Generator(tf.keras.Model):

    def __init__(self):
        """
        implementation of Generator
        :param input_size: size of the sequence (input noise)
        """
        super().__init__(name='generator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Dense(units=HIDDEN_UNITS, input_shape=()))  ##  input_shape =
        self.model.add(Reshape((-1, HIDDEN_UNITS, SEQ_LENGTH)))

        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())

        self.conv1 = tf.keras.layers.Conv1D(filters=N_CHAR, kernel_size=1)

    def __call__(self, inputs):
        x = self.model(inputs)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = softmax(x)
        return x


class Discriminator(tf.keras.Model):

    def __init__(self, clip = 1):
        """
        implementation of Discriminator
        :param clip: value to which you clip the gradients (or False)
        """
        super().__init__(name='discriminator')
        self.residual = tf.keras.models.Sequential([ResidualBlock(), ResidualBlock(), ResidualBlock(),
                                                    ResidualBlock(), ResidualBlock()])
        self.conv1d = Conv1D(N_CHAR, HIDDEN_UNITS, 1)
        self.linear = Dense(SEQ_LENGTH*HIDDEN_UNITS, 1)


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

