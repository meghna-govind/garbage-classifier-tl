import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image # Strreamlit works with PIL library very easily for Images
import cv2
st.title("Image - CLassifier")
upload = st.file_uploader('Label=Upload the image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB) # Color from BGR to RGB
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)
  model = keras.models.load_model('/content/gdrive/MyDrive/Transfer_learning.hdf5')
  x = cv2.resize(opencv_image,(224,224))
  x = np.expand_dims(x,axis=0)
  x = preprocess_input(x)
  y = model.predict(x)
  classes=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
  label = classes[np.argmax(y)]
  st.title(label)