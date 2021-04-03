from utils import load_test_train, load_all, simple_blur, image_preprocess_cnns, load_raw_image
from blurer import gen_blur_df
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from keras import backend as K
from config import simpleCNN_model_dir, blur_images_path
import cv2
import numpy as np
import matplotlib.pyplot as plt

test_list, _ = load_test_train()

# print(test_list[:5])

simple_cnn = keras.models.load_model(simpleCNN_model_dir)

def simple_cnn_wrapper(images):
    tmp_imgs = []
    for image in images:
        tmp_imgs.append(image_preprocess_cnns(image))
    tmp_imgs = np.array(tmp_imgs)
    # print(tmp_imgs.shape)

    return simple_cnn.predict_on_batch(tmp_imgs).reshape(-1)
    
df = gen_blur_df(
    test_list,
    range(250),
    load_raw_image,
    simple_cnn_wrapper,
    simple_blur
)

df.head()

df.to_csv(blur_images_path + '/test_simple_cnn.csv')
   