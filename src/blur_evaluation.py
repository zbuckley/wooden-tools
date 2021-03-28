from utils import load_test_train, load_all
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from config import simpleCNN_model_dir, vggCNN_model_dir
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

test_list, _ = load_test_train()

# print(test_list[:5])

x_test, y_test, _ = load_all(test_list)

def simple_blur(image, window_size):
    return cv2.blur(image, (window_size, window_size))

def gaussian_blur(image, window_size):
    # setting stddev to 0 implies cv2 will calculate based on window_size
    return cv2.GaussianBlur(image, (window_size, window_size), 0)

simple_cnn = keras.models.load_model(simpleCNN_model_dir)
vgg_cnn = keras.models.load_model(vggCNN_model_dir)

def get_blurs_acc(blurs, blur_fnc, model):
    # start from unblurred images... blur_size of 0
    #   then apply blur of increasing sizes... 
    results = []
    for blur_size in blurs:
        # print(blur_size)
        # apply blur to all images
        if blur_size is 0:
            tmp_imgs = x_test
        else:
            tmp_imgs = np.zeros(x_test.shape)
            # print(tmp_imgs.shape) 

            # load tmp_imgs with blurred images
            for i in range(x_test.shape[0]):
                tmp_imgs[i, :, :, :] = blur_fnc(
                    x_test[i, :, :, :],
                    blur_size
                )

        # perform prediction using model
        preds = model(tmp_imgs)
        preds = np.round(preds)
        results.append(accuracy_score(y_test, preds))

    return list(blurs), results

x, y = get_blurs_acc(range(0, 250, 1), simple_blur, simple_cnn)

plt.plot(x, y)
plt.title('Accuracy vs Blur Window Size')
plt.xlabel('Blur Window Size (Pixels)')
plt.ylabel('Classification Accuracy')
plt.savefig('blur_accuracy_simple_cnn')


x, y = get_blurs_acc(range(0, 250, 1), simple_blur, simple_cnn)

plt.plot(x, y)
plt.title('Accuracy vs Blur Window Size')
plt.xlabel('Blur Window Size (Pixels)')
plt.ylabel('Classification Accuracy')
plt.savefig('blur_accuracy_simple_cnn')

    