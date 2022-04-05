import os
import sys
import asyncio
import warnings
import numpy as np
import pickle as pkl
from fastapi import FastAPI
from PIL import Image, ImageFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, Response
from keras.applications.imagenet_utils import preprocess_input
from utils.tf_functions import (
  AnimeClassifier,
  process_image_tf,
  load_image_embeddings
)

EVA_INNER = 3
SIZE_IMG = 224
EVA_CLASS = 32
EVA_UNITS = 1024
EVA_TYPE  = 'resnet'
EVA_MODEL_CLASS = 32
#Shape of vector result: (2048,)

class_array_eva = pkl.load(open( f'./data/class_array_{EVA_CLASS}.pkl', 'rb'))

parmas_eval = {
  'units': EVA_UNITS,
  'inner_layers': EVA_INNER,
  'type_extractor': EVA_TYPE,
  'num_classes':  EVA_MODEL_CLASS,
  'input_shape': (SIZE_IMG, SIZE_IMG, 3)
}
model_eva = AnimeClassifier(**parmas_eval)
model_eva.build(input_shape=(None, *parmas_eval['input_shape']))
PATH_BEST_EVA = f'./models/{EVA_TYPE}_{EVA_MODEL_CLASS}class_{EVA_UNITS}_units_{EVA_INNER}.h5'
model_eva.load_weights(PATH_BEST_EVA)

indexer_gpu, data_image, vector_images = load_image_embeddings(entire_db = True, factor = 16)


warnings.filterwarnings('ignore')
app = FastAPI()


app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*']
)

def search_result(image):
  k = 20
  vector_search = model_eva.vectorize(np.array([image])).numpy()
  _, ids_result = indexer_gpu.search(vector_search, k)

  result = set()
  current_id = 1
  for id_v in ids_result[0]:
    class_name = data_image[id_v][1]
    result.add(class_name)
  return list(result)

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
  print('===========================================')
  print(f'Image: {image.filename}')
  
  image = process_image_tf(image.file, SIZE_IMG)
  data = search_result(image)
  return {
    'data': data
  }