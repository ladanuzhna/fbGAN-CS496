from Gan import GAN
from Models import FeedbackNet
from globals import *
from utils.protein_utilities import protein_to_DNA, DNA_to_protein

class GAN_FBNet():

    def __init__(self, generator_path=None, discriminator_path=None, fbnet_path=None):
        self.GAN = GAN(generator_weights_path=generator_path, discriminator_weights_path=discriminator_path)
        self.FBNet = FeedbackNet()

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
            self.data += best_samples

        return best_samples

    def train(self, inputs, epochs):
        self.data = inputs

        for epoch in range(epochs):
            dataset = self.create_dataset(self.data)

            for sample_batch in dataset:
                G_loss = self.GAN.G_train_step()
                D_loss, GP = self.GAN.D_train_step(sample_batch)

                generated = self.GAN.G.generate_samples(number=BATCH_SIZE, decoded=True)
                scores = get_scores(genreated)
                best_samples = self.add_samples(generated, scores)

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset
