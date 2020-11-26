import tensorflow as tf
from Gan import GAN
from utils.protein_utilities import protein_to_DNA
from utils.data_utilities import get_sequences, OneHot_Seq

# Load the pretrained feedback net
feedback = tf.keras.models.load_model('blah')
generator = Models.Generator()
discriminator = Models.Discriminator()

# Training loop


###############################################################################################
# GAN training example with DNA sequences (obtained by translating the protein sequences)

# Load protein sequences and shuffle them
path = './data/<something>'
X_train, X_test, y_train, y_test = get_sequences(path, split=0.01)
X_train = X_train.tolist()

print(f'Number of training samples: {len(X_train)}')
np.random.shuffle(X_train)

# Translate to DNA encoding
X = protein_to_DNA(X_train)
print(f'Example of translated DNA sequences: {X[:5]}')

# One Hot encode into 5 categories, ATCG and P for padded positions
OneHot = OneHot_Seq(letter_type='DNA')
real_sequences = OneHot.seq_to_onehot(X)

print(f'OneHot encoding of DNA sequences: {real_sequences[:5]}')

# Train WGAN-GP
gan = GAN(discriminator_steps=0, lr=0.0002, gradient_penalty_weight=5)
gan.train(real_sequences, epochs=10, step_log=25)
gan.plot_history()
gan.show_sequences_history()
