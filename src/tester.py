# let's test the windowing function
from segmenter import window, heatmap
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/mnt/c/Users/buckl/My Documents/GitHub/wooden-tools/data/Images/Cou1NoUselarge0.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('tmp.png')

print(img.shape)

def fake_model_fnc(image): 
    return np.random.uniform(0, 1)


result = window(
    img, 
    fake_model_fnc,
    (100, 100),
    5, 5
)

print(result.shape)

heatmap(img, result, alpha=0.3)
plt.savefig('tmp2.png')
