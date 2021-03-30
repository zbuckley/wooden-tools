from utils import load_test_train, load_all, get_blurs_acc, simple_blur
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from config import vggCNN_model_dir, blur_images_path
import cv2
import numpy as np
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)
# ])

test_list, _ = load_test_train()

x_test, y_test, _ = load_all(test_list)

vgg_cnn = keras.models.load_model(vggCNN_model_dir)

x, y = get_blurs_acc(range(0, 250, 1), simple_blur, vgg_cnn, x_test, y_test, K)

plt.plot(x, y)
plt.title('Accuracy vs Blur Window Size')
plt.xlabel('Blur Window Size (Pixels)')
plt.ylabel('Classification Accuracy')
plt.savefig(blur_images_path + '/blur_accuracy_vgg_cnn')
   