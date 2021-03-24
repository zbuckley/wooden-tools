import tensorflow as tf
from tensorflow import keras
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from config import vggCNN_model_dir, simpleCNN_model_dir, full_images_dir, segmentation_images_dir
from segmenter import window, heatmap
import cv2
import numpy as np
import matplotlib.pyplot as plt

simple_cnn = keras.models.load_model(simpleCNN_model_dir)
vgg_cnn = keras.models.load_model(vggCNN_model_dir)

images_list = [
    ((896, 896), 'Cou1', 'Cou1/COU1.tif'),
    ((889, 889), 'Cou10', 'Cou10/LostTech_ChimpTools-10_02032020.jpg'),
    ((904, 904), 'Cou10-nodamage', 'Cou10/LostTech_ChimpTools-10_no_damage_02032020.jpg'),
    ((909, 909), 'Cou11a', 'Cou11a/LostTech_ChimpTools-11a_02032020.jpg'),
    ((898, 898), 'Cou11a-nodamage', 'Cou11a/LostTech_ChimpTools-11a_no_damage_02032020.jpg'),
    ((890, 890), 'Cou11b', 'Cou11b/LostTech_ChimpTools-11b_02032020.jpg'),
    ((902, 902), 'Cou11b-nodamage', 'Cou11b/LostTech_ChimpTools-11b_no_damage_02032020.jpg'),
    ((890, 890), 'Cou12', 'Cou12/LostTech_ChimpTools-12d_02032020.jpg'),
    ((889, 889), 'Cou12-nodamage', 'Cou12/LostTech_ChimpTools-12_no_damage_02032020.jpg'),
    ((900, 900), 'Cou13', 'Cou13/LostTech_ChimpTools-13b_02032020.jpg'),
    ((900, 900), 'Cou13-nodamage', 'Cou13/LostTech_ChimpTools-13_no_damage_02032020.jpg'),
    ((905, 905), 'Cou14', 'Cou14/LostTech_ChimpTools-14_02032020.jpg'),
    ((908, 908), 'Cou14-nodamage', 'Cou14/LostTech_ChimpTools-14_no_damage_02032020.jpg'),
    ((902, 902), 'Cou2', 'Cou2/Cou2_20181113.png'),
    ((885, 885), 'Cou4', 'Cou4/Cou4_20181113.png'),
    ((899, 899), 'Cou5', 'Cou5/Cou5_20181113.png'),
    ((880, 880), 'Cou6', 'Cou6/COU6_17_12_2018.png'),
    ((890, 890), 'Cou6b', 'Cou6b/COU6b_17_12_2018.png'),
    ((878, 878), 'Cou7', 'Cou7/COU_7_18_12_2018.png'),
]

window_size, out_name, img_path = images_list[0]
img_path = full_images_dir + '/' + img_path

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
        # print('Now Processing...', img_path)
        build_overlaid_image(model, window_size, out_name + '-' + model_name, img_path, 25)