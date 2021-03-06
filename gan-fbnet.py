import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import os

from gan import GAN
from globals import *
from utils.protein_utilities import protein_to_DNA, DNA_to_protein


class GAN_FBNet():

    def __init__(self, generator_path=None, discriminator_path=None,
                 fbnet_path=None, features=DESIRED_FEATURES):
        self.GAN = GAN(generator_weights_path=generator_path, discriminator_weights_path=discriminator_path)
        self.FBNet = tf.keras.models.load_model(fbnet_path)

        _, _, _, _, self.tokenizer = get_data_for_feedback(PATH_DATA)
        self.label_order = np.array(['B', 'C', 'E', 'G', 'H', 'I', 'S',
                                     'T'])  # order of labels as output by Multilabel binarizer - don't change!
        self.desired_features = features
        self.data = None
        self.checkpoint_dir = './'

    def get_scores(self, inputs):
        # convert the DNA sequences to protein sequences
        protein_sequence = DNA_to_protein(inputs)
        input_grams = triplets(protein_sequence)
        transformed = self.tokenizer.texts_to_sequences(input_grams)
        transformed = sequence.pad_sequences(transformed, maxlen=MAX_LEN, padding='post')
        # use FBNet to grade the sequences
        scores = self.FBNet.predict(transformed)
        return scores

    def get_score_per_feature(self, scores):
        scores = np.array(scores)
        avg_scores = np.rint(100 * np.mean(scores, axis=0))
        score_per_feature = []
        for feature in self.desired_features:
            i = int(np.where(feature == self.label_order)[0])
            score_i = int(avg_scores[i])
            fscore = (feature, score_i)
            score_per_feature.append(fscore)

        return score_per_feature

    def add_samples(self, generated, scores, score_threshold=0.75, replace=False):
        best_index = scores > score_threshold
        best_samples = []
        best_scores = []
        for i in range(len(best_index)):
            passed_threshold = set(self.label_order[best_index[i]])
            if set(self.desired_features).issubset(passed_threshold):
                best_samples.append(generated[i])
                best_scores.append(scores[i])

        if replace:
            pass
        else:
            self.data = np.concatenate((self.data, np.array(best_samples)), axis=0)
        return best_samples, best_scores

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
                OneHot = OneHot_Seq(letter_type=TASK_TYPE)
                decoded_generated = OneHot.onehot_to_seq(generated)
                scores = self.get_scores(decoded_generated)
                generated = tf.cast(generated, tf.float32)
                best_samples, best_scores = self.add_samples(generated, scores)

                self.data = np.concatenate((self.data, best_samples), axis=0)

                if step % step_log == 0:
                    print(
                        f'\tStep {step}:   Generator: {G_loss.numpy()}   Discriminator: {D_loss.numpy()}   Samples: {len(self.data)}')

                    print('\tBest scores per feature: ', end=' ')
                    sc = self.get_score_per_feature(best_scores)
                    pprint = [f'{sc[0]}: {sc[1]}%' for sc in sc]
                    print(*pprint, sep=' ')

                    print('Average scores per feature: ', end=' ')
                    sc = self.get_score_per_feature(scores)
                    pprint = [f'{sc[0]}: {sc[1]}%' for sc in sc]
                    print(*pprint, sep=' ')

                if step == 100:
                    break

                step += 1

            percent_fake = int((len(self.data) - len(inputs)) / len(self.data) * 100)
            print(f'\tPercent of the fake samples in the discriminator: {percent_fake}%.')

    def create_dataset(self, inputs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.shuffle(inputs.shape[0], seed=0).batch(self.batch_size, drop_remainder=True)
        return dataset