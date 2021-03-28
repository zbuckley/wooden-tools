

root_dir='C:/Users/buckl/Documents/GitHub/wooden-tools'

data_dir=root_dir + '/data'
models_dir=root_dir + '/models'

simpleCNN_model_dir = models_dir + '/SimpleCNN'
vggCNN_model_dir = models_dir + '/VGGCNN'

segmentation_images_dir = data_dir + '/segmentation_images'
full_images_dir = data_dir + '/FullImages'
images_dir = data_dir + '/Images'

train_list_path = data_dir + '/' + 'train_large.csv'

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