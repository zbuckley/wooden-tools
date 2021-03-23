import numpy as np
import cv2
import matplotlib.pyplot as plt

# windowing loop
  # window_sizes 754x754, 1058x1058; always square
def window (
    original_img, # BGR; loaded by opencv
    model_fnc, # function for preprocessing, and returning probability
    window_size, # size of the desired window
    row_step,
    col_step
):
    # maximum x and y size of the image
    row_max, col_max, channels = original_img.shape
    print('row_max', row_max)
    print('col_max', col_max)
    print('channels', channels)

    window_row, window_col = window_size

    # initiate accumlator
    #  since we know how large the resulting array should be, this should allow quicker
    #  accumulation than python lists
    num_row = int(row_max/row_step)
    num_col = int(col_max/col_step)
    accum = np.zeros((num_row, num_col))

    for row in range(num_row):
        for col in range(num_col):
            # print('Accum Row:', y, 'Accum Col:', x)

            img_row = row*row_step
            img_col = col*col_step
            # print('Image Row:', img_y, 'Image Col:', img_x)

            img_slice = original_img[(img_row):(img_row + window_row), (img_col):(img_col + window_col), :]
            # print(x, y, window_x, window_y)
            # print(img_slice.shape)
            accum[row, col] = model_fnc(img_slice)

    return accum

def heatmap(original_img, matrix, alpha=0.4):
    print(original_img.shape)
    print(matrix.shape)
    orig_rows, orig_cols, _ = original_img.shape
    overlay = cv2.resize(matrix, (orig_cols, orig_rows))
    print('A', overlay.shape)
    overlay = cv2.applyColorMap(np.uint8(overlay*255), cv2.COLORMAP_JET)
    
    print(original_img.shape)
    print(overlay.shape)

    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), alpha=alpha)
