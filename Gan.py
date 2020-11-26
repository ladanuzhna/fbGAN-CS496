import tensorflow as tf
import matplotlib.pyplot as plt

from Models import Generator, Discriminator

import globals 


# BATCH_SIZE = 32
# NOISE_SHAPE = 128


class GAN():

    def __init__(self, batch_size=BATCH_SIZE, discriminator_steps=5, lr=0.0002, gradient_penalty_weight=10):
        self.batch_size = batch_size
        self.G = Generator()
        self.D = Discriminator()
        self.d_steps = discriminator_steps

        self.history = {"G_losses": [], "D_losses": [], "gradient_penalty": [], "sequences": []}

        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

        self.gp_weight = gradient_penalty_weight

    def generate_samples(self, number=None):
        if number is None:
            number = self.batch_size
        z = tf.random.normal([number, NOISE_SHAPE])
        generated = self.G(z)
        return generated

    def generator_loss(self, fake_score):
        return -tf.math.reduce_mean(fake_score)

    def discriminator_loss(self, real_score, fake_score):
        return tf.math.reduce_mean(real_score) - tf.math.reduce_mean(fake_score)

    # @tf.function
    def gradient_penalty(self, real_samples, fake_samples):
        alpha = tf.random.normal([self.batch_size, 1, 1], 0.0, 1.0)
        real_samples = tf.cast(real_samples, tf.float32)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.D(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    # @tf.function
    def G_train_step(self):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            fake_score = self.D(fake_samples, training=True)
            G_loss = self.generator_loss(fake_score)

        G_gradients = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients((zip(G_gradients, self.G.trainable_variables)))

        return G_loss

    # @tf.function
    def D_train_step(self, real_samples):
        with tf.GradientTape() as tape:
            fake_samples = self.generate_samples()
            real_score = self.D(real_samples, training=True)
            fake_score = self.D(fake_samples, training=True)

            D_loss = self.discriminator_loss(real_score, fake_score)
            GP = self.gradient_penalty(real_samples, fake_samples) * self.gp_weight
            D_loss = D_loss + GP

        D_gradients = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients((zip(D_gradients, self.D.trainable_variables)))

        return D_loss, GP

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset

    def train(self, inputs, epochs):

        # Pre-train discriminator 
        for step in range(self.d_steps):
            dataset = self.create_dataset(inputs).as_numpy_iterator()

            for sample_batch in dataset:
                self.D_train_step(sample_batch)

        # Train discriminator and generator
        for epoch in range(epochs):
            dataset = self.create_dataset(inputs).as_numpy_iterator()

            print(f"Epoch {epoch}/{epochs}:")

            for sample_batch in dataset:
                G_loss = self.G_train_step()
                D_loss, GP = self.D_train_step(sample_batch)

            example_sequence = gan.generate_samples(number=1)

            self.history["G_losses"].append(G_loss.numpy())
            self.history["D_losses"].append(D_loss.numpy())
            self.history['gradient_penalty'].append(GP.numpy())
            self.history['sequences'].append(example_sequence)

            print(f"\tGenerator loss: {round(G_loss.numpy(), 2)}. \tDiscriminator loss: {round(D_loss.numpy(), 2)}")

    def plot_history(self):
        D_losses = np.array(self.history['D_losses'])
        G_losses = np.array(self.history['G_losses'])

        plt.plot(np.arange(D_losses.shape[0]), D_losses, label='Discriminator loss')
        plt.plot(np.arange(G_losses.shape[0]), G_losses, label='Generator loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        plt.show()

    def show_sequences_history(self):
        OneHot = OneHot_Seq()
        sequences_history = self.history['sequences']
        decod = [OneHot.onehot_to_seq(seq.numpy()) for seq in sequences_history]

        print('History of generated sequences... \n')
        for i in range(len(decod)):
            print(f'Epoch {i}: \t {decod[i][0]}')
