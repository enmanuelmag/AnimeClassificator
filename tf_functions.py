import gc
import faiss
import numpy as np
import random as rd
import pickle as pkl
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras import backend as k
from tensorflow.keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers

SEED = 42
#LEN_TF_RECORD = 11753
LEN_TF_RECORD = 4007
SUFIX = '_faces' if True else ''
PATH_VECTORS = f'./data/data_image_{LEN_TF_RECORD}{SUFIX}.pkl'

rd.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


class ReleaseMemory(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()
    k.clear_session()


class AddPositionEmbs(layers.Layer):
  """
  Inputs are image patches Custom layer to add positional embeddings to the inputs.
  tf.keras.initializers.RandomNormal(stddev=0.02)
  """
  def __init__(self, posemb_init=None, **kwargs):
    super(AddPositionEmbs, self).__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)
    return inputs + pos_embedding

def mlp_block_f(mlp_dim, inputs):
  x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
  x = layers.Dropout(0.1)(x) # dropout rate is from original paper,
  x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x) # check GELU paper
  x = layers.Dropout(0.1)(x)
  return x

def EncoderBlock(inputs, num_heads, mlp_dim):
  x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x) 
  # self attention multi-head, dropout_rate is from original implementation
  x = layers.Add()([x, inputs]) # 1st residual part 
  
  y = layers.LayerNormalization(dtype=x.dtype)(x)
  y = mlp_block_f(mlp_dim, y)
  y_1 = layers.Add()([y, x]) #2nd residual part 
  return y_1

def Encoder(inputs, num_layers, mlp_dim, num_heads):
  x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
  x = layers.Dropout(0.2)(x)
  for _ in range(num_layers):
    x = EncoderBlock(x, num_heads, mlp_dim)

  encoded = layers.LayerNormalization(name='EncoderNorm')(x)
  return encoded

def generate_patch_conv_orgPaper_f(inputs, patch_size, hidden_size):
  patches = tf.keras.layers.Conv2D(
    filters=hidden_size,
    kernel_size=patch_size,
    strides=patch_size, #the stride is the same as the patch (piece) size
    padding='valid'
  )(inputs)
  row_axis, col_axis = (1, 2) # channels last images
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x

def ViTModel(class_types, transformer_layers, patch_size, hidden_size, num_heads, mlp_dim, shape=(224, 224, 3)):
  rescale_layer = tf.keras.Sequential([layers.Rescaling(1./255)])
  inputs = layers.Input(shape=shape)
  
  rescale = rescale_layer(inputs) # rescaling (normalizing pixel val between 0 and 1)
 
  patches = generate_patch_conv_orgPaper_f(rescale, patch_size, hidden_size) # generate patches with conv layer

  encoder_out = Encoder(patches, transformer_layers, mlp_dim, num_heads) # ready for the transformer blocks

  #  final part (mlp to classification)
  im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)

  logits = layers.Dense(
    units=class_types,
    name='Head',
    kernel_initializer=tf.keras.initializers.zeros
  )(im_representation) # !!! important !!! activation is linear 
  return tf.keras.Model(inputs=inputs, outputs=logits)



class AnimeClassifier(tf.keras.Model):
  def __init__(self, num_classes, input_shape, units=1024, inner_layers=12, type_extractor='vgg'):
    assert type_extractor in ['vgg', 'inception', 'resnet']
    assert inner_layers >= 1
    assert num_classes >= 8
    assert len(input_shape) == 3
    assert units >= 64

    super(AnimeClassifier, self).__init__(name='AnimeClassifier')

    self.units = units
    self.in_layer = tf.keras.layers.Input(input_shape, name='input')

    if type_extractor == 'vgg':
      feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=self.in_layer)
    elif type_extractor == 'inception':
      feature_extractor = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=self.in_layer)
    elif type_extractor == 'resnet':
      feature_extractor = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=self.in_layer)
    else:
      raise ValueError('type_extractor must be vgg, inception or resnet')

    self.feature_extractor = feature_extractor
    self.global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
    self.flatten = tf.keras.layers.Flatten()

    self.hidden_mlp = []
    for i in range(inner_layers):
      self.hidden_mlp.append(tf.keras.layers.Dense(units,activation=tf.nn.relu))
      self.hidden_mlp.append(tf.keras.layers.Dropout(0.5, seed=SEED))

    self.out_layer = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)

  def call(self, inputs, training=None, mask=None):
    x = self.feature_extractor(inputs, training=training)
    x = self.global_average_pooling(x)
    x = self.flatten(x, training=training)
    for layer in self.hidden_mlp:
      x = layer(x, training=training)
    return self.out_layer(x, training=training)

  def predict_class(self, x):
    return tf.argmax(self(x), axis=1)

  def vectorize(self, x, flatten=True):
    x = self.feature_extractor(x)
    x = self.global_average_pooling(x)
    if flatten:
      return self.flatten(x)
    return x


def parse_tfrecord_vec(tfrecord, size):
  x = tf.io.parse_single_example(tfrecord, {
    'image': tf.io.FixedLenFeature([], tf.string),
    'class_name': tf.io.FixedLenFeature([], tf.string),
  })
  x_train = tf.image.decode_jpeg(x['image'], channels=3)
  x_train = tf.image.resize(x_train, (size, size))
  x_train = preprocess_input(x_train, mode='tf')

  y_train = x['class_name']
  if y_train is None:
    y_train = ''

  return x_train, y_train

def load_tfrecord_dataset_vec(file_pattern, size):
  files = tf.data.Dataset.list_files(file_pattern)
  dataset = files.flat_map(tf.data.TFRecordDataset)
  return dataset.map(lambda x: parse_tfrecord_vec(x, size))

def parse_record_vec(combination):
  item_1, item_2 = combination
  img_1, label_1 = item_1
  img_2, label_2 = item_2
  return (img_1, img_2, label_1 == label_2)

def load_image_embeddings(entire_db = False, factor = 16):
  data_image = pkl.load(open(PATH_VECTORS, 'rb'))
  vector_images = np.array(list(data_image[:, 0])) 

  d = 2048 #Shape of vector result: (2048,)
  nb = LEN_TF_RECORD
  res = faiss.StandardGpuResources()  # use a single GPU

  # build a flat (CPU) index
  index_flat = faiss.IndexFlatL2(d)
  # make it into a gpu index
  indexer_gpu = faiss.index_cpu_to_gpu(res, 0, index_flat)
  indexer_gpu.add(vector_images)
  
  print(f'All vectors: {indexer_gpu.ntotal}')
  return indexer_gpu, data_image, vector_images

def process_image_tf(image, size):
  print('Image byte', image)
  #image = tf.io.read_file(image_path)
  #image = tf.image.decode_jpeg(image, channels=3)
  image = Image.open(image).convert("RGB")
  #image = tf.image.decode_jpeg([ image ], channels=3)[0]
  image = tf.image.resize(image, (size, size))
  return preprocess_input(image, mode='tf')






class GeneratePatch(tf.keras.layers.Layer):
  def __init__(self, patch_size):
    super(GeneratePatch, self).__init__(name='GeneratePatch')
    self.patch_size = patch_size

  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
      images=images, 
      sizes=[1, self.patch_size, self.patch_size, 1], 
      strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID"
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
    return patches

class PatchEncodeEmbed(layers.Layer):
  """
  Positional Encoding Layer, 2 steps happen here
    1. Flatten the patches
    2. Map to dim D; patch embeddings
  """
  def __init__(self, num_patches, projection_dim):
    super(PatchEncodeEmbed, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    return self.projection(patch) + self.position_embedding(positions)
