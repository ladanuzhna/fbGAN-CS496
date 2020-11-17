import tensorflow as tf
from Models import Generator, Discriminator

BATCH_SIZE = 32
NOISE_SHAPE = 128

class GAN():

    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.G = Generator()
        self.D = Discriminator()

        self.G_loss = generator_loss
        self.D_loss = discriminator_loss

        self.history = {"G_losses": [], "D_losses": []}

        # TODO: Add WGAN lipschitz-penalty

    def compile(self, lr=1e-4):
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # .minimize(self.G_loss, var_list=self.G_list)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # .minimize(self.D_loss,var_list=self.D_list)

    def generate_samples(self):
        z = tf.random.normal([self.batch_size, NOISE_SHAPE])
        generated = self.G(z)
        return generated

    @tf.function
    def G_train_step(self):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            fake_score = self.D(fake_samples, training=True)
            G_loss = self.G_loss(fake_score)

        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients((zip(G_gradients, self.G.trainable_variables)))

        return G_loss

    @tf.function
    def D_train_step(self, real_samples):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            real_score = self.D(real_samples, training=True)
            fake_score = self.D(fake_samples, training=True)

            D_loss = discriminator_loss(real_score, fake_score)

        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients((zip(D_gradients, self.D.trainable_variables)))

        return D_loss

    def create_dataset(self, input):
        dataset = tf.data.Dataset.from_tensor_slices(input)
        dataset = dataset.shuffle(input.shape[0], seed=0).batch(self.batch_size)
        return dataset

    def train(self, input, epochs):
        # TODO: Print out losses per i steps correctly

        dataset = self.create_dataset(input)

        for epoch in range(epochs):

            print(f"Epoch {epoch}/{epochs}:")

            for sample_batch in dataset:
                G_loss = self.G_train_step()
                D_loss = self.D_train_step(sample_batch)

            self.history["G_losses"].append(G_loss)
            self.history["D_losses"].append(D_loss)

            print(f"\tGenerator loss: {G_loss}. \tDiscriminator loss: {D_loss}")


def generator_loss(fake):
    return -tf.math.reduce_mean(fake)

def discriminator_loss(real, fake):
    return tf.math.reduce_mean(real) - tf.math.reduce_mean(fake)
