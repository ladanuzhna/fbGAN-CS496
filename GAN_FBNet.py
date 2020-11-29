from Gan import GAN
from Models import FeedbackNet

class GAN_FBNet():

    def __init__(self, generator_path=None, discriminator_path=None):
        self.GAN = GAN(generator_weights_path = generator_path, discriminator_weights_path = discriminator_path)
        self.FBNet = FeedbackNet()

        self.data = None

        self.checkpoint_dir = './'

    def get_scores(self):
        # use FBNet to grade the sequences
        pass

    def pick_best(self):
        pass

    def add_samples(self, replace = False):
        # add best sequences to the self.data, shuffle
        pass

    def train(self, inputs):
        self.data = inputs





