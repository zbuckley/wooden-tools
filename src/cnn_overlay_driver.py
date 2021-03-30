import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from config import vggCNN_model_dir, simpleCNN_model_dir, full_images_dir, segmentation_images_dir, images_list
from segmenter import window, heatmap
import cv2
import numpy as np
import matplotlib.pyplot as plt

simple_cnn = keras.models.load_model(simpleCNN_model_dir)
vgg_cnn = keras.models.load_model(vggCNN_model_dir)

def build_model_fnc(model):
    def model_fnc(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        pred = model(np.array([image]))
        # print('PREDICTION:', pred)
        return pred
    return model_fnc
        
# model = build_model_fnc(simple_cnn)
# print(model)
# print(type(model))

def build_overlaid_image(model, window_size, output_name, image_path, step_size):
    print(image_path)
    image = cv2.imread(image_path)
    result = window(
        image, 
        build_model_fnc(model),
        window_size,
        step_size, step_size
    )
    img_y, img_x, _ = image.shape
    crop_y = int(window_size[0]/2)
    crop_x = int(window_size[1]/2)
    cropping = image[crop_y:(img_y - crop_y), crop_x:(img_x - crop_x)]
    heatmap(cropping, result, alpha=0.4)
    plt.savefig(segmentation_images_dir + '/' + output_name)

for model_name, model in [('cnn', simple_cnn), ('vgg', vgg_cnn)]:
    for window_size, out_name, img_path in images_list:
        img_path = full_images_dir + '/' + img_path
        build_overlaid_image(model, window_size, out_name + '-' + model_name, img_path, 10)