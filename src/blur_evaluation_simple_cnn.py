from utils import load_test_train, load_all, get_blurs_acc, simple_blur
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)
])
from tensorflow import keras
from keras import backend as K
from config import simpleCNN_model_dir, blur_images_path
import cv2
import numpy as np
import matplotlib.pyplot as plt

test_list, _ = load_test_train()

# print(test_list[:5])

x_test, y_test, _ = load_all(test_list)

simple_cnn = keras.models.load_model(simpleCNN_model_dir)

x, y = get_blurs_acc(range(0, 250, 1), simple_blur, simple_cnn, x_test, y_test, K)

plt.plot(x, y)
plt.title('Accuracy vs Blur Window Size')
plt.xlabel('Blur Window Size (Pixels)')
plt.ylabel('Classification Accuracy')
plt.savefig(blur_images_path + '/blur_accuracy_simple_cnn')
   