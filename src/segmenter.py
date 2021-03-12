import numpy as np

# windowing loop
  # window_sizes 754x754, 1058x1058; always square
def window (
    original_img, # BGR; loaded by opencv
    model_fnc, # function for preprocessing, and returning probability
    window_size, # size of the desired window
    x_step,
    y_step
):
    # we'll need the x and y components of the window
    #   TODO: this is really a placeholder, necessary because i'm not sure 
    #       if i'm right about 0 being width, and 1 being height
    window_x = np.round(window_size[0]/2) # TODO: is 0 the width?
    window_y = np.round(window_size[1]/2) # TODO: is 1 the height?

    # we'll iterate on the center points, so we'll need to figure what our first point is.
    x_0 = window_x
    y_0 = window_y
    
    # maximum x and y size of the image
    x_max, y_max, channels = original_img.shape

    # initiate accumlator
    #  since we know how large the resulting array should be, this should allow quicker
    #  accumulation than python lists
    accum = np.zeros((
        np.floor((y_max - 2*window_y)/(window_y + y_step)),
        np.floor((x_max - 2*window_x)/(window_x + x_step))
    ))

    for y in range(y_0, y_step, y_max):
        for x in range(x_0, x_step, x_max):
            print('Row:', y, 'Col:', x)
            img_slice = original_img[(y - window_y):(y + window_y), (x - window_x):(x + window_x)]
            accum[x, y] = model_fnc(img_slice)

    return accum


# def heatmap(original_img, matrix):
#     # plot both images