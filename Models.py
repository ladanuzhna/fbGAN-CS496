import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Dense, Reshape, ReLU, Permute

SEQ_LENGTH = 50
DIM = 16
KERNEL_SIZE = 5
BATCH_SIZE = None
N_CHAR = 20
N_SAMPLES = 2000
NOISE_SHAPE = 128

def softmax(logits):
    shape = tf.shape(logits)
    res = tf.nn.softmax(tf.reshape(logits, [-1,N_CHAR]))
    return tf.reshape(res, shape)

class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.relu = ReLU()
        self.conv1d_1 = Conv1D(filters=DIM, kernel_size=KERNEL_SIZE, padding='same', strides=1, activation='relu')
        self.conv1d_2 = Conv1D(filters=DIM, kernel_size=KERNEL_SIZE, padding='same', strides=1)

    def __call__(self,X,alpha = 0.3):
        x = self.relu(X)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        return x + alpha*x

class Generator(tf.keras.Model):

    def __init__(self):
        """
        implementation of Generator
        :param input_size: size of the sequence (input noise)
        """
        super().__init__(name='generator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape = (NOISE_SHAPE,), batch_size = BATCH_SIZE))
        self.model.add(Dense(units = DIM*SEQ_LENGTH))
        self.model.add(Reshape((-1, DIM, SEQ_LENGTH)))

        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())

        self.conv1 = Conv1D(filters = N_CHAR, kernel_size = 1)

    def __call__(self, inputs):
        x = self.model(inputs)
        x = softmax(x)
        return x

class Discriminator(tf.keras.Model):

    def __init__(self, clip = 1):
        """
        implementation of Discriminator
        :param clip: value to which you clip the gradients (or False)
        """
        super().__init__(name='discriminator')

        self.model = tf.keras.models.Sequential()
        self.model.add(Input(shape = (SEQ_LENGTH,N_CHAR), batch_size = BATCH_SIZE))
        self.model.add(Conv1D(filters = DIM, kernel_size = 1))

        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())
        self.model.add(ResidualBlock())

        self.model.add(Reshape((-1,DIM*SEQ_LENGTH)))
        self.model.add(Dense(units = DIM*SEQ_LENGTH))
        self.model.add(Dense(units = 1))


    def __call__(self,inputs,training = False):
        """
        model's forward pass
        :param X: input of the size [batch_size, seq_length];
        :param training: specifies the behavior of the call;
        :return: Y: probability of each sequences being real of shape [batch_size, 1]
        """

        x = self.model(inputs)
        return x



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

