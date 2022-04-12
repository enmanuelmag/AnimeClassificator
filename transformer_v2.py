import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MLP(layers.Layer):
  def __init__(self, hidden_units, dropout_rate, **kwargs):
    self.len_hidden_units = len(hidden_units)
    super(MLP, self).__init__(**kwargs)
    for i, units in enumerate(hidden_units):
      setattr(self, f'dense_{i}', layers.Dense(units, activation=tf.nn.gelu))
      setattr(self, f'dropout_{i}', layers.Dropout(dropout_rate))

  def call(self, x, training=None):
    for i in range(self.len_hidden_units):
      x = getattr(self, f'dense_{i}')(x)
      x = getattr(self, f'dropout_{i}')(x, training=training)
    return x

class Patches(layers.Layer):
  def __init__(self, patch_size):
    super(Patches, self).__init__(name='Patches')
    self.patch_size = patch_size

  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
      images=images,
      sizes=[1, self.patch_size, self.patch_size, 1],
      strides=[1, self.patch_size, self.patch_size, 1],
      rates=[1, 1, 1, 1],
      padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches

class PatchEncoder(layers.Layer):
  def __init__(self, num_patches, projection_dim):
    super(PatchEncoder, self).__init__(name='PatchEncoder')
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
      input_dim=num_patches, output_dim=projection_dim
    )

  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) + self.position_embedding(positions)
    return encoded


class VitModel(keras.Model):
  def __init__(
    self, transformer_layers, patch_size, num_patches, projection_dim,
    num_heads, mlp_head_units, num_classes, transformer_units, dropout_rate=0.1
  ):
    super(VitModel, self).__init__(name='VitModel')
    self.patch_size = patch_size
    self.num_patches = num_patches
    self.projection_dim = projection_dim
    self.transformer_units = transformer_units
    self.num_heads = num_heads
    self.mlp_head_units = mlp_head_units
    self.dropout_rate = dropout_rate
    self.transformer_layers = transformer_layers

    self.patches = Patches(patch_size)
    self.patch_encoder = PatchEncoder(num_patches, projection_dim)

    for i in range(transformer_layers):
      setattr(self, f'norm_{i}', layers.LayerNormalization(epsilon=1e-6, name=f'norm_{i}'))
      setattr(self, f'multihead_{i}', layers.MultiHeadAttention(num_heads, key_dim=projection_dim, dropout=0.1, name=f'multihead_{i}'))
      setattr(self, f'add_{i}', layers.Add(name=f'add_{i}'))
      setattr(self, f'norm_add_{i}', layers.LayerNormalization(epsilon=1e-6, name=f'norm_add_{i}'))
      setattr(self, f'mlp_{i}', MLP(transformer_units, dropout_rate, name=f'mlp_{i}'))
      setattr(self, f'encoded_patchs_{i}', layers.Add(name=f'encoded_patchs_{i}'))

    self.representation = layers.LayerNormalization(epsilon=1e-6, name='representation')
    self.flatten = layers.Flatten(name='flatten_representation')
    self.drop_representation = layers.Dropout(0.5, name='dropout_representation')

    self.features = MLP(mlp_head_units, dropout_rate=0.5, name='features')
    self.logits = layers.Dense(num_classes, name='logits')
  
  def call(self, inputs, training=None):
    patches = self.patches(inputs)
    encoded_patches = self.patch_encoder(patches)
    for i in range(self.transformer_layers):
      x1 = getattr(self, f'norm_{i}')(encoded_patches)
      attention_output = getattr(self, f'multihead_{i}')(x1, x1)
      x2 = getattr(self, f'add_{i}')([attention_output, encoded_patches])
      x3 = getattr(self, f'norm_add_{i}')(x2)
      x3 = getattr(self, f'mlp_{i}')(x3)
      encoded_patches = getattr(self, f'encoded_patchs_{i}')([x3, x2])
    
    representation = self.representation(encoded_patches)
    representation = self.flatten(representation)
    representation = self.drop_representation(representation, training=training)
    features = self.features(representation)
    return self.logits(features)

  """Cumstom function for personal propurses"""	
  def expand(self, input_shape):
    x = layers.Input(shape=input_shape[1:], name='inputs')
    return tf.keras.Model(inputs=[x], outputs=self.call(x))

  def vectorize(self, inputs, training=False, flatten=False):
    patches = self.patches(inputs)
    encoded_patches = self.patch_encoder(patches)
    for i in range(self.num_heads):
      x1 = getattr(self, f'norm_{i}')(encoded_patches)
      attention_output = getattr(self, f'multihead_{i}')(x1, x1)
      x2 = getattr(self, f'add_{i}')([attention_output, encoded_patches])
      x3 = getattr(self, f'norm_add_{i}')(x2)
      x3 = getattr(self, f'mlp_{i}')(x3)
      encoded_patches = getattr(self, f'encoded_patchs_{i}')([x3, x2])
    representation = self.representation(encoded_patches)
    if flatten:
      representation = self.flatten(representation)
    return representation