import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time

SEED = 42

class AddPositionEmbs(layers.Layer):
  def __init__(self, posemb_init=None):
    super(AddPositionEmbs, self).__init__(name='add_position_embs')
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)
    return inputs + pos_embedding

class MLPBlock(layers.Layer):
  def __init__(self, mlp_dim):
    super(MLPBlock, self).__init__(name='mlp_block')
    self.mlp_dim = mlp_dim
    self.dense_1 = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)
    self.dropout_1 = layers.Dropout(0.1, seed=SEED)
    self.dense_2 = layers.Dense(units=64, activation=tf.nn.gelu) #shape[-1]
    self.dropout_2 = layers.Dropout(0.1, seed=SEED)

  def call(self, inputs):
    x = self.dense_1(inputs)
    x = self.dropout_1(x)
    x = self.dense_2(x)
    x = self.dropout_2(x)
    return x

class EncoderBlock(layers.Layer):
  def __init__(self, num_heads, mlp_dim):
    super(EncoderBlock, self).__init__(name='encoder_block')
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads

    self.normalization_1 = layers.LayerNormalization()
    #self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)
    self.multi_head_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=512, dropout=0.1)
    self.add_1 = layers.Add()
    self.add_2 = layers.Add()
    self.normalization_2 = layers.LayerNormalization()
    self.mlp_block = MLPBlock(mlp_dim)

  def call(self, inputs):
    x = self.normalization_1(inputs)
    x = self.multi_head_attention(x, x)
    x = self.add_1([x, inputs])
    y = self.normalization_2(x)
    y = self.mlp_block(y)
    y_1 = self.add_2([y, x])
    return y_1

class Encoder(layers.Layer):
  def __init__(self, num_layers, mlp_dim, num_heads, dropout=0.2):
    super(Encoder, self).__init__(name='encoder')
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.num_layers = num_layers

    self.position_embs = AddPositionEmbs(
      posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02)
    )
    self.dropout = layers.Dropout(dropout, seed=SEED)
    for i in range(num_layers):
      setattr(self, f'encoder_block_{i}', EncoderBlock(num_heads, mlp_dim))

    self.normalization = layers.LayerNormalization()
    
  def call(self, inputs):
    x = self.position_embs(inputs)
    x = self.dropout(x)
    for i in range(self.num_layers):
      x = getattr(self, f'encoder_block_{i}')(x)
    x = self.normalization(x)
    return x

class PatchConv(layers.Layer):
  def __init__(self, patch_size, hidden_size):
    super(PatchConv, self).__init__(name='patch_conv')
    self.patch_size = patch_size
    self.hidden_size = hidden_size
    self.conv = layers.Conv2D(
      filters=self.hidden_size,
      kernel_size=self.patch_size,
      strides=self.patch_size, #the stride is the same as the patch (piece) size
      padding='valid'
    )
    self.row_axis, self.col_axis = (1, 2) # channels last images

  def call(self, inputs):
    patches = self.conv(inputs)
    seq_len = (inputs.shape[self.row_axis] // self.patch_size) * (inputs.shape[self.col_axis] // self.patch_size)
    x = tf.reshape(patches, [-1, seq_len, self.hidden_size])
    return x


class ViTModel(tf.keras.Model):
  def __init__(self, class_types, transformer_layers, patch_size, hidden_size, num_heads, mlp_dim, shape=(224, 224, 3)):
    super(ViTModel, self).__init__(name='ViTModel')
    self.rescale_layer = layers.Rescaling(1./255, name='rescale_layer')
    #self.rescale_layer = tf.keras.Sequential([layers.Rescaling(1./255, name='rescale_layer')])
    self.patch_conv_layer = PatchConv(patch_size, hidden_size)
    self.encoder = Encoder(transformer_layers, mlp_dim, num_heads)
    self.out_layer = layers.Dense(
      units=class_types,
      kernel_initializer=tf.keras.initializers.zeros,
      activation=tf.nn.softmax,
      name='out_layer'
    )

  """ def build(self, input_shape):
    result = layers.Input(shape=input_shape[1:], name='inputs')
    result = self.patch_conv_layer(result)
    result = self.encoder(result)
    result = self.out_layer(result) """
  
  def call(self, x):
    x = self.rescale_layer(x)
    x = self.patch_conv_layer(x)
    x = self.encoder(x)
    x = tf.math.reduce_mean(x, axis=1, name='reduce_mean')
    x = self.out_layer(x)
    return x

  """Cumstom function for personal propurses"""	
  def expand(self, input_shape):
    x = layers.Input(shape=input_shape[1:], name='inputs')
    return tf.keras.Model(inputs=[x], outputs=self.call(x))
  
  def predict_class(self, x):
    return tf.argmax(self.call(x), axis=1)

  def vectorize(self, x, reduce=True):
    x = self.rescale_layer(x)
    x = self.patch_conv_layer(x)
    x = self.encoder(x)
    if reduce:
      return tf.math.reduce_mean(x, axis=1, name='reduce_mean')
    return x
