import pandas as pd
from config import train_list_path, images_dir
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize


def load_test_train():
    def filter_name(imagePath):
        return 'large' in imagePath and 'largest' not in imagePath
 
    train_list_file = pd.read_csv(train_list_path)
    tmp_train_list = train_list_file['path'].str.split('/').apply(lambda x: x[-1])
    # print(tmp_train_list[:5])
    train_dict = {k:'NOT FOUND' for k in tmp_train_list.tolist()}
    
    imagePaths = list(paths.list_images(images_dir))
    # print(images_dir)
    # print(imagePaths[:5])

    largeImages = [] #accumulators for full large image paths (both train and test)
    for imagePath in imagePaths:
        if filter_name(imagePath):
            largeImages.append(imagePath)

    # print(largeImages[:5])
    del imagePaths # cuz why not?

    # for each path with image name matching; set full path for image in dictionary
    test_list = [] # accumulator for full large image path (test set only)
    for imagePath in largeImages:
        fileName = os.path.basename(imagePath)
        # print(fileName)
        if fileName in train_dict:
            train_dict[fileName] = imagePath
        else:
            test_list.append(imagePath)

    return test_list, train_dict.values()


def isUsed(imagePath):
    if 'NoUse' in imagePath:
        return 'NoUse'
    else:
        return 'Use'

def load_raw_image(imagePath):
    label = label_binarize([isUsed(imagePath)], classes=['NoUse', 'Use'])[0][0]
    image = cv2.imread(imagePath)
    return (label, image)

def image_preprocess_cnns(bgr_image):
    image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (224, 224))

# returns (label, image)
def load_image(imagePath):
    label = isUsed(imagePath)
    image = cv2.imread(imagePath)
    # TODO: is it ok that BGR2RGB converion occurs before resize?
    #   also ensure all functions understand that conversion to RGB has already occurred...
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return (label, image)

def load_all(imagePaths, lb = None):
  data = []
  labels = []

  for imagePath in imagePaths:
    print('.', end='')
    label, image = load_image(imagePath)
    data.append(image)
    labels.append(label)

  labels = np.array(labels)
  data = np.array(data)

  #TODO: confirm this makes sense... maybe just use label_binarize
  if lb is None:
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
  else:
    labels = lb.transform(labels)

  return data, labels, lb

## NOTE: DEPRECATED; will be replaced when blurer.py is working
def get_blurs_acc(blurs, blur_fnc, model, x_test, y_test, K):
    # start from unblurred images... blur_size of 0
    #   then apply blur of increasing sizes... 
    results = []
    for blur_size in blurs:
        # print(blur_size)
        # apply blur to all images
        if blur_size == 0:
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
        preds = np.zeros((tmp_imgs.shape[0], 1))
        for i in range(0, int(tmp_imgs.shape[0]/32)):
            start_idx = i*32
            end_idx = min((i+1)*32, tmp_imgs.shape[0])
            preds[start_idx:end_idx, :] = model.predict_on_batch(tmp_imgs[start_idx:end_idx, :, :, :])
        
        preds = np.round(preds)
        results.append(accuracy_score(y_test, preds))
        # K.clear_session()

    return list(blurs), results

def simple_blur(image, window_size):
    if window_size == 0:
        return image
    else:
        return cv2.blur(image, (window_size, window_size))

def gaussian_blur(image, window_size):
    # from documentation setting stddev to 0 implies cv2 will calculate based on window_size
    return cv2.GaussianBlur(image, (window_size, window_size), 0)

