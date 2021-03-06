# GLOBAL VARIABLES
import os

DESIRED_FEATURES = ['C', 'H', 'E']

MAX_LEN = 243
MIN_LEN = 0

TASK_TYPE = 'DNA'
MAX_LEN_PROTEIN = MAX_LEN // 3 - 6
MIN_LEN_PROTEIN = MIN_LEN // 3 - 6

SEQ_LENGTH = MAX_LEN
DIM = 50
KERNEL_SIZE = 5
BATCH_SIZE = 128
N_CHAR = 5
NOISE_SHAPE = 128



###### UPDATE THE PATHS ######

# Select a path where your data are stores
PATH_DATA = '/drive/MyDrive/Colab Notebooks/FB-GAN_496/data/2018-06-06-ss.cleaned.csv'

# Select paths to the saved weights of the gan and feedback
PATH_GAN = '/drive/MyDrive/Colab Notebooks/FB-GAN_496/weights/gan'
PATH_G = os.path.join(PATH_GAN, 'G243')
PATH_D = os.path.join(PATH_GAN, 'D243')

PATH_FB = '/drive/MyDrive/Colab Notebooks/FB-GAN_496/weights/feedback'
