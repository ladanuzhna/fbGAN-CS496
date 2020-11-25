import tensorflow as tf
import Models


#Load the pretrained feedback net
feedback = tf.keras.models.load_model('blah')
generator = Models.Generator()
discriminator = Models.Discriminator()

#Training loop


################################################################################
# A very simple GAN training example
BATCH_SIZE = 32

path = 'data/2018-06-06-ss.cleaned.csv'  # change path
X_train,X_test,y_train,y_test = get_data(path, return_sequence=True)

X = X_train[:BATCH_SIZE]
OneHot = OneHot_Seq()
real_sequences = OneHot.seq_to_onehot(X)

gan = GAN()
gan.compile()
gan.train(real_sequences, epochs = 2)