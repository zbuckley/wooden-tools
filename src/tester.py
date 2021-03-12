# let's test the windowing function
from segmenter import window
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("/mnt/c/Users/buckl/My Documents/GitHub/wooden-tools/data/Images/Cou1NoUselarge0.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.savefig('tmp.png')

result = window(
    img, 
    lambda x: .8,
    (100, 100),
    100, 100
)