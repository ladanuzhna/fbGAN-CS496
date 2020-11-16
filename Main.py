import tensorflow as tf
import Models

#Load the pretrained feedback net
feedback = tf.keras.models.load_model('blah')
generator = Models.Generator()
discriminator = Models.Discriminator()

#Training loop