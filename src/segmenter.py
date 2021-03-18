import numpy as np
import cv2
import matplotlib.pyplot as plt

# windowing loop
  # window_sizes 754x754, 1058x1058; always square
def window (
    original_img, # BGR; loaded by opencv
    model_fnc, # function for preprocessing, and returning probability
    window_size, # size of the desired window
    x_step,
    y_step
):
    # maximum x and y size of the image
    x_max, y_max, channels = original_img.shape

    window_x, window_y = window_size

    # initiate accumlator
    #  since we know how large the resulting array should be, this should allow quicker
    #  accumulation than python lists
    num_x = int(x_max/x_step)
    num_y = int(y_max/y_step)
    accum = np.zeros((num_y, num_x))

    for y in range(num_y):
        for x in range(num_x):
            # print('Accum Row:', y, 'Accum Col:', x)

            img_y = y*y_step
            img_x = x*x_step
            # print('Image Row:', img_y, 'Image Col:', img_x)

            img_slice = original_img[(img_y):(img_y + window_y), (img_x):(img_x + window_x), :]
            accum[x, y] = model_fnc(img_slice)

    return accum

def heatmap(original_img, matrix, alpha=0.7):
    orig_x, orig_y, _ = original_img.shape
    overlay = cv2.resize(matrix, (orig_x, orig_y))
    overlay = cv2.applyColorMap(np.uint8(overlay*255), cv2.COLORMAP_JET)
    print(overlay.shape)

    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), alpha=alpha)
