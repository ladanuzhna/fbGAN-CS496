import tensorflow as tf
from Models import Generator, Discriminator


class GAN():

    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()

        self.G_cost = generator_loss()
        self.D_cost = discriminator_loss()

        self.D_list = []
        self.G_list = []

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.D_list]

    def compile(self):
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.G_cost,
                                                                                                       var_list=self.G_list)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.D_cost,
                                                                                                       var_list=self.D_list)

    def train_iter_G(self):
        pass

    def train_iter_D(self):
        pass

    def train(self):
        pass



def generator_loss(fake):
    return -tf.math.reduce_mean(fake)

def discriminator_loss(real, fake):
    return tf.math.reduce_mean(real) - tf.math.reduce_mean(fake)