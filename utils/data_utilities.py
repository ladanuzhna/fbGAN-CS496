import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

MAX_LEN = 128  # maximum length of the sequence
MIN_LEN = 40

def triplets(sequences):
    """
    Apply sliding window of length 3 to each sequence in the input list
    :param sequences: list of sequences
    :return: numpy array of triplets for each sequence
    Usage: Split protein sequence into triplets of aminoacids
    """
    return np.array([[aminoacids[i:i + 3] for i in range(len(aminoacids))] for aminoacids in sequences])


def get_data(return_sequence=False):
    """

    :param return_sequence: parameter that specifies whether we want to return raw sequence or its embedding
    :return: X_train,X_test,Y_train,Y_test
    """
    df = pd.read_csv('Data/protein_structure.csv')

    """
    Columns of interest for classification:
    seq: the sequence of the peptide
    sst8: the eight-state (Q8) secondary structure
    sst3: the three-state (Q3) secondary structure
    len: the length of the peptide
    has_nonstd_aa: whether the peptide contains nonstandard amino acids (B, O, U, X, or Z).
    """
    input_seqs, target_seqs = df[['seq', 'sst8']][(df.len <= MAX_LEN) & (~df.has_nonstd_aa)].values.T

    # Transform features
    tokenizer_encoder = Tokenizer()
    input_grams = triplets(input_seqs)
    tokenizer_encoder.fit_on_texts(input_grams)
    input_data = tokenizer_encoder.texts_to_sequences(input_grams)
    input_data = sequence.pad_sequences(input_data, maxlen=MAX_LEN, padding='post')

    # Transform targets
    tokenizer_decoder = Tokenizer(char_level=True)
    tokenizer_decoder.fit_on_texts(target_seqs)
    target_data = tokenizer_decoder.texts_to_sequences(target_seqs)
    target_data = sequence.pad_sequences(target_data, maxlen=MAX_LEN, padding='post')
    target_data = to_categorical(target_data)

    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=.3, random_state=1)
    seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.3,
                                                                      random_state=1)
    if return_sequence:
        return seq_train, seq_test, target_train, target_test
    else:
        return X_train, X_test, y_train, y_test




def get_sequences(path=None, min_len=MIN_LEN, max_len=MAX_LEN):
    if path is None:
        path = '/content/drive/My Drive/Colab Notebooks/FB-GAN_496/data/2018-06-06-ss.cleaned.csv'

    df = pd.read_csv(path)
    input_seqs, target_seqs = df[['seq', 'sst8']][
        (df.len >= min_len) & (df.len <= max_len) & (~df.has_nonstd_aa)].values.T
    seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.3, random_state=1)

    return seq_train, seq_test, target_train, target_test


def get_dataset(sequences, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices(sequences)
    dataset = dataset.shuffle(sequences.shape[0], seed=0).batch(batch_size)
    return dataset


class OneHot_Seq:
    def __init__(self, amino_acids=None, max_length=MAX_LEN):
        """
        :param amino_acids: list of amino acid letters
        :param max_length: maximum length of a protein sequence
        """

        if amino_acids is None:
            self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        else:
            self.amino_acids = amino_acids

        self.amino_acids_dict = {f'{aa}': i + 1 for i, aa in enumerate(self.amino_acids)}
        self.invert_dict = {v: k for k, v in self.amino_acids_dict.items()}
        self.invert_dict[0] = '0'

        self.max_length = max_length

    def _parse_pad_sequences(self, sequences):
        """
        Parse strings into sequences encoded with integers 1-21 for amino acids or 0 if padded
        """
        parse = lambda seq: np.array([a for a in seq])
        parsed = pd.DataFrame(sequences).iloc[:, 0].apply(parse)

        for i in range(parsed.shape[0]):
            parsed[i] = np.vectorize(self.amino_acids_dict.get)(parsed[i])

        parsed = pad_sequences(parsed, maxlen=self.max_length, value=0, padding='post')

        return parsed

    def seq_to_onehot(self, sequences):
        """
        Return an array of one-hot encodings from amino acid strings.
        :param sequences: ndarray of strings, shape = (N,1) where N is the number of samples
        :return: array of onehot encoded sequences, shape = (N, max_length, amino_acids)
        """
        sequences = self._parse_pad_sequences(sequences)
        onehot = []

        for seq in sequences:
            onehot_seq = np.zeros((seq.size, len(self.amino_acids) + 1))
            onehot_seq[np.arange(seq.size), seq] = 1
            onehot.append(onehot_seq)

        return np.array(onehot)

    def onehot_to_seq(self, sequences):
        """
        Returns an array of strings from one-hot encoding.
        :param sequences: ndarray of shape (N, max_length, amino_acids) where N is the number of samples
        :return: array of strings of shape (N, 1)
        """
        sequences = np.argmax(sequences, axis=2)
        sequences = np.vectorize(self.invert_dict.get)(sequences)
        decoded_sequences = [[''.join([aa for aa in seq])] for seq in sequences]

        return decoded_sequences
