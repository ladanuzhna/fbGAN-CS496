import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from utils.protein_utilities import protein_to_DNA
from globals import *


global n_words, MAX_LEN_FB
MAX_LEN_FB = 128  # length of max sequence we want to consider for training
n_tags = 8  # number of classes in 8-state prediction


def triplets(sequences):
    """
    Apply sliding window of length 3 to each sequence in the input list
    :param sequences: list of sequences
    :return: numpy array of triplets for each sequence
    Usage: Split protein sequence into triplets of aminoacids
    """
    return np.array([[aminoacids[i:i + 3] for i in range(len(aminoacids))] for aminoacids in sequences])


def get_data_for_feedback(path=PATH_DATA, max_len = MAX_LEN_FB):
    df = pd.read_csv(path)
    input_seqs, target_seqs = df[['seq', 'sst8']][(df.len <= max_len) & (~df.has_nonstd_aa)].values.T

    # Transform features
    input_data, tokenizer = transform_sequence(input_seqs)
    # Transform targets
    mlb = MultiLabelBinarizer()
    target_data = mlb.fit_transform(target_seqs)

    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=.3, random_state=1)
    return X_train, X_test, y_train, y_test, tokenizer

def transform_sequence(seqs, tokenizer_encoder = None):
  # transforms sequences for input into feedback net, tokenizes + adds padding
  # if there is no given tokenizer_encoder -> initialize one and fit it on a given sequence
  # o.w. just transform the sequence with given tokenizer
  # returns transformed sequences + tokenizer that was fit on the input dataset
  if not tokenizer_encoder:
    tokenizer_encoder = Tokenizer()
    input_grams = triplets(seqs)
    tokenizer_encoder.fit_on_texts(input_grams)
  transformed = tokenizer_encoder.texts_to_sequences(input_grams)
  transformed = sequence.pad_sequences(transformed, maxlen=MAX_LEN, padding='post')
  return transformed,tokenizer_encoder

#
# def get_data(return_sequence=False):
#     """
#
#     :param return_sequence: parameter that specifies whether we want to return raw sequence or its embedding
#     :return: X_train,X_test,Y_train,Y_test
#     """
#     df = pd.read_csv('Data/protein_structure.csv')
#
#     """
#     Columns of interest for classification:
#     seq: the sequence of the peptide
#     sst8: the eight-state (Q8) secondary structure
#     sst3: the three-state (Q3) secondary structure
#     len: the length of the peptide
#     has_nonstd_aa: whether the peptide contains nonstandard amino acids (B, O, U, X, or Z).
#     """
#     input_seqs, target_seqs = df[['seq', 'sst8']][(df.len <= MAX_LEN) & (~df.has_nonstd_aa)].values.T
#
#     # Transform features
#     tokenizer_encoder = Tokenizer()
#     input_grams = triplets(input_seqs)
#     tokenizer_encoder.fit_on_texts(input_grams)
#     input_data = tokenizer_encoder.texts_to_sequences(input_grams)
#     input_data = keras.preprocessing.sequence.pad_sequences(input_data, maxlen=MAX_LEN, padding='post')
#
#     # Transform targets
#     tokenizer_decoder = Tokenizer(char_level=True)
#     tokenizer_decoder.fit_on_texts(target_seqs)
#     target_data = tokenizer_decoder.texts_to_sequences(target_seqs)
#     target_data = keras.preprocessing.sequence.pad_sequences(target_data, maxlen=MAX_LEN, padding='post')
#     target_data = to_categorical(target_data)
#
#     X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=.3, random_state=1)
#     seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.3,
#                                                                       random_state=1)
#     if return_sequence:
#         return seq_train, seq_test, target_train, target_test
#     else:
#         return X_train, X_test, y_train, y_test


def get_sequences(path=PATH_DATA, min_len=MIN_LEN_PROTEIN, max_len=MAX_LEN_PROTEIN):

    df = pd.read_csv(path)
    input_seqs, target_seqs = df[['seq', 'sst8']][
        (df.len >= min_len) & (df.len <= max_len) & (~df.has_nonstd_aa)].values.T
    seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.3,
                                                                      random_state=1)

    return seq_train, seq_test, target_train, target_test


def get_dataset(sequences, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.shuffle(sequences.shape[0], seed=0).batch(batch_size)
    return dataset


def parse(sequences):
    if type(sequences) == str:
        parsed = np.array([a for a in sequences])
        return parsed

    parse = lambda seq: np.array([a for a in seq])
    parsed = pd.DataFrame(sequences).iloc[:, 0].apply(parse).to_numpy().tolist()

    return parsed


def prepare_dataset(path =PATH_DATA, split=0.01):
    # Load protein sequences and shuffle them
    X_train, _, _, _ = get_sequences(path, split)
    X_train = X_train.tolist()
    np.random.shuffle(X_train)

    # print(f'Number of training samples: {len(X_train)}')

    # Translate to DNA encoding
    X = protein_to_DNA(X_train)
    # print(f'Example of translated DNA sequences: \n {X[0]}')

    # One Hot encode into 5 categories, ATCG and P for padded positions
    OneHot = OneHot_Seq(letter_type='DNA')
    real_sequences = OneHot.seq_to_onehot(X)

    # print(f'Example of OneHot encoding of DNA sequences: {real_sequences[0]}')

    return real_sequences


class OneHot_Seq:
    def __init__(self, letter_type='DNA', letters=None, max_length=MAX_LEN):
        """
        :param letter_type: str 'amino acids' or 'DNA'. If a different type is used, provide custom letters.
        :param max_length: int maximum length of a sequence. Sequences will be padded to this length.
        """

        if letter_type == 'amino acids':
            self.letters = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                            'W', 'Y', 'V']

        elif letter_type == 'DNA':
            self.letters = ['A', 'T', 'C', 'G']

        else:
            assert letters is not None
            self.letters = letters

        self.letters_dict = {f'{aa}': i + 1 for i, aa in enumerate(self.letters)}
        self.invert_dict = {v: k for k, v in self.letters_dict.items()}
        self.invert_dict[0] = 'P'

        self.max_length = max_length

    def _parse_pad_sequences(self, sequences):

        parse = lambda seq: np.array([a for a in seq])
        parsed = pd.DataFrame(sequences).iloc[:, 0].apply(parse)

        for i in range(parsed.shape[0]):
            parsed[i] = np.vectorize(self.letters_dict.get)(parsed[i])

        parsed = keras.preprocessing.sequence.pad_sequences(parsed, maxlen=self.max_length, value=0, padding='post')

        return parsed

    def seq_to_onehot(self, sequences):
        """
        Return an array of one-hot encodings from sequence strings.
        :param sequences: ndarray of strings, shape = (N,1) where N is the number of samples
        :return: array of onehot encoded sequences, shape = (N, max_length, amino_acids)
        """
        sequences = self._parse_pad_sequences(sequences)
        onehot = []

        for seq in sequences:
            onehot_seq = np.zeros((seq.size, len(self.letters) + 1))
            onehot_seq[np.arange(seq.size), seq] = 1
            onehot.append(onehot_seq)

        return np.array(onehot)

    def onehot_to_seq(self, sequences):
        """
        Returns an array of strings from one-hot encoding.
        :param sequences: ndarray of shape (N, max_length, amino_acids) where N is the number of samples
        :return: array of strings of shape (N, 1)
        """
        if sequences.ndim == 2:
            sequences = np.argmax(sequences, axis=1)
            sequences = np.vectorize(self.invert_dict.get)(sequences)
            decoded_sequences = [''.join([aa for aa in sequences])]
            return decoded_sequences

        sequences = np.argmax(sequences, axis=2)
        sequences = np.vectorize(self.invert_dict.get)(sequences)
        decoded_sequences = [[''.join([aa for aa in seq])] for seq in sequences]

        return decoded_sequences
