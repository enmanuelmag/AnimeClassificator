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
EVA_UNITS = 2048 #2048 1024
EVA_TYPE  = 'resnet'
EVA_MODEL_CLASS = 32
SUFIX = '_faces' if True else ''
#Shape of vector result: (2048,)

class_array_eva = pkl.load(open(f'./data/class_array{"_faces" if True else ""}_{EVA_CLASS}.pkl', 'rb'))
print(class_array_eva)

parmas_eval = {
  'units': EVA_UNITS,
  'inner_layers': EVA_INNER,
  'type_extractor': EVA_TYPE,
  'num_classes':  EVA_MODEL_CLASS,
  'input_shape': (SIZE_IMG, SIZE_IMG, 3)
}
model_eva = AnimeClassifier(**parmas_eval)
model_eva.build(input_shape=(None, *parmas_eval['input_shape']))
PATH_BEST_EVA = f'./models/{EVA_TYPE}_{EVA_MODEL_CLASS}class_{EVA_UNITS}_units_{EVA_INNER}{SUFIX}.h5'
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

  #save image on temp folder
  #image_path = f'./temp/{image.filename}'
  #with open(image_path, 'wb') as f:
  #  f.write(image.file.read())
  image = process_image_tf(image.file, SIZE_IMG)
  print(image.shape)
  data = search_result(image)
  return {
    'data': data
  }

@app.post("/predict/name")
def predict_name(image: UploadFile = File(...)):
  print('===========================================')
  print(f'Image: {image.filename}')

  image = process_image_tf(image.file, SIZE_IMG)
  print(image.shape)
  result = model_eva.predict(np.array([image]))[0]
  class_name = class_array_eva[np.argmax(result)]
  print(f'Class: {class_name}', np.argmax(result))
  return {
    'data': class_name
  }