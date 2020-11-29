from Gan import GAN
from Models import FeedbackNet
from globals import *
from utils.protein_utilities import protein_to_DNA, DNA_to_protein


class GAN_FBNet():

    def __init__(self, generator_path=None, discriminator_path=None, fbnet_path=None):
        self.GAN = GAN(generator_weights_path=generator_path, discriminator_weights_path=discriminator_path)
        self.FBNet = None  # FeedbackNet()

        self.data = None

        self.checkpoint_dir = './'

    def get_scores(self, inputs):
        # convert the DNA sequences to protein sequences
        protein_sequence = DNA_to_protein(inputs)
        # use FBNet to grade the sequences
        scores = None

        return scores

    def add_samples(self, generated, scores, score_threshold=0.75, replace=False):
        best_index = scores > score_threshold
        best_samples = generated[best_index]
        if replace:
            pass
        else:
            self.data = np.concatenate((self.data, generated), axis=0)

        return best_samples

    def train(self, inputs, epochs, step_log=50, steps_per_epoch=100, batch_size=BATCH_SIZE):
        self.data = inputs
        self.batch_size = BATCH_SIZE

        for epoch in range(epochs):
            dataset = self.create_dataset(self.data)

            print(f'Epoch {epoch} / {epochs}')

            step = 0

            for sample_batch in dataset:
                G_loss = self.GAN.G_train_step()
                D_loss, GP = self.GAN.D_train_step(sample_batch)

                generated = self.GAN.generate_samples(number=BATCH_SIZE, decoded=False)
                generated = tf.cast(generated, tf.float32)
                # scores = get_scores(genreated)
                # best_samples = self.add_samples(generated, scores)

                self.data = np.concatenate((self.data, generated), axis=0)

                if step % step_log == 0:
                    print(
                        f'\tStep {step}:   Generator: {G_loss.numpy()}   Discriminator: {D_loss.numpy()}   Samples: {len(self.data)}')

                if step == 100:
                    break

                step += 1

            percent_fake = int((len(self.data) - len(inputs)) / len(self.data) * 100)
            print(f'\tPercent of the fake samples in the discriminator: {percent_fake}%.')

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset

