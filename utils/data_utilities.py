import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

MAX_LEN = 128  # maximum length of the sequence

def triplets(sequences):
    """
    Apply sliding window of length 3 to each sequence in the input list
    :param sequences: list of sequences
    :return: numpy array of triplets for each sequence
    Usage: Split protein sequence into triplets of aminoacids
    """
    return np.array([[aminoacids[i:i+3] for i in range(len(aminoacids))] for aminoacids in sequences])

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
    seq_train, seq_test, target_train, target_test = train_test_split(input_seqs, target_seqs, test_size=.3,random_state=1)
    if return_sequence:
        return seq_train,seq_test,target_train,target_test
    else:
        return X_train,X_test,y_train,y_test