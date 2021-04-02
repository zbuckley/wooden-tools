import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score


# class intended for internal use by the gen_blur_df function
#   class wraps behaviour of processing the images to ensure we can process them
#   in chunks, matching the batch_size parameter
class BlurProcessingManager:
    def __init__(self, model_fnc, batch_size):
        self.batch_contents = {}
        self.batch_size = batch_size
        self.model_fnc = model_fnc
        self.results = None

    def __init_batch_contents(self):
        self.batch_contents = {}
        self.batch_contents['images'] = []
        self.batch_contents['labels'] = []
        self.batch_contents['image_paths'] = []
        self.batch_contents['blur_sizes'] = []

    def __add_batch_record(self, image, label, image_path, blur_size):
        self.batch_contents['images'].append(image)
        self.batch_contents['labels'].append(label)
        self.batch_contents['image_paths'].append(image_path)
        self.batch_contents['blur_sizes'].append(blur_sizes)

    def __collect_results(self):
        del self.batch_contents['images']
        if self.results is None:
            self.results = pd.DataFrame.from_dict(self.batch_contents)
        else:
            self.results = self.results.append(self.batch_contents, ignore_index=True)
        del self.batch_contents
        self.__init_batch_contents()
            
    def process(self, image, label, image_path, blur_size):
        if len(self.batch_contents) < self.batch_size:
            self.__add_batch_record(image, label, image_path, blur_size)
        else:
            self.__run_batch_contents()
            
    def run_batch_contents(self):
        # perform predictions
        preds = self.model_fnc(self.batch_contents['images'])

        # store results in batch_contents dict
        self.batch_contents['predictions'] = preds

        # reset for next batch
        self.__collect_results()


# function for generating a dataframe of blur response data
#  Note: could return all blur responses... from blurred images... 
#    seems to be overkill for now... but may give opportunity for 
#    better backtracing/analysis later
#  params
#  blurs, the iterators for the blur_size values to use
#  image_load_fnc, function for loading image from path, should return label and image as tuple (in that order)
#  model_fnc, the function for performing preprocessing, and retrieving predictions
#    model_fnc should expect to recieve a list of images, the length of the list will match batch_size
#    NOTE: model_fnc is expected to return a 1 for used, and 0 for not used
#  image paths
def gen_blur_df(img_paths, blurs, load_image_fnc, model_fnc, blur_fnc, batch_size=32):
    # initialize manager class instance
    bp = BlurProcessingManager(model_fnc, batch_size)

    for image_path in img_paths:
        label, image = load_image_fnc(image_path)
        bp.process(image, label, image_path, 0)
        for blur_size in blurs:
            tmp_image = blur_fnc(image, blur_size)
            bp.process(tmp_image, label, image_path, blur_size)

    # after processing all images, need to ensure process manager flushes the remainings 
    #  images so we don't leave any unprocessed.
    bp.run_batch_contents()

    return bp.results


## DEPRECATED; will be remove when gen_blur_df is confirmed to work
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
    return cv2.blur(image, (window_size, window_size))

def gaussian_blur(image, window_size):
    # from documentation setting stddev to 0 implies cv2 will calculate based on window_size
    return cv2.GaussianBlur(image, (window_size, window_size), 0)