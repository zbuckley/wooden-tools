# Wooden Tools

## Setup

Instructions for setting up repo and necessary data sets

### Data Directory

Cloning the repository should have included the data directory, and segmentation_images directory. 

First download the following datasets from various google drive locations, and extract the resulting zip file into the `data` directory:

* https://drive.google.com/drive/folders/18M_0S-oo7BNW__7SWpcGA0EuQGkMAVh1?usp=sharing  (FullImages)
* https://drive.google.com/drive/folders/1MewoGzSxqbkH3gqg5oFDypvcHnDUlBe7?usp=sharing  (Images)

Note: Access to this location must be granted, prior to these links working.

If this works, you should have `data/FullImages`, and `data/Images` directories.

Last, you'll need to download the `train_large.csv` file from: 

* https://drive.google.com/drive/folders/1MewoGzSxqbkH3gqg5oFDypvcHnDUlBe7?usp=sharing


### Models Directory

Similarly the Models directory will have also been created from cloning the repository. 

To run any scripts that make use of the tensorflow based neural networks, you will need to download the pre-trained models from various google drive locations:

* https://drive.google.com/drive/folders/1zx5V8dKeGKNDBbxmOl2LJfHbL6eRPMMW?usp=sharing (SimpleCNN)
* https://drive.google.com/drive/folders/1-N7iKHu6gbG6PHAVhJLj9nfRwRZw-TT3?usp=sharing (VGGCNN)

Note: Access to this location must be granted, prior to these links working.

After the model definitions are extracted from the zip files you should have `models/SimpleCNN`, and `models/VGGCNN` direcotories in your local repo.

### Segmentation Images Directory

This directory is intended to contain resulting data from the segmentation evaluation process.

My initial attempt at segmenting the data based on the code in the repo can be downloaded here:

* https://gwu.box.com/s/22rik4vzfj279pre9qhud9f5p1l6yfps

Note: Access to this location must be granted, prior to these links working.

After these files are downloaded and extracted, you should a number of images in the `data/segmentation_images` directory. These images follow a naming convention: 

* `<original image name>-<model abbreviation>.png`

where `<model abbreviation>` is one of:

* `cnn` - referring to our simple CNN model
* `vgg` - referring to our VGG CNN model

### Blur Images Directory

This directory is intended to contain resulting data from the blur evaluation process.

My initial attempt at generating these plots based on the neural neteworks (only contains vgg at this time):

* https://gwu.box.com/s/jgjpthredw7qp0br6in2oovk8rmgaed2

### Update config.py

The `config.py` data is the sole source of directory locations, and other data inputs used by other scripts within the repository. 


## Code

This section contains brief descriptions of the scripts found in the `src/` directory of the repository.

### Segmentation Images

The segmentation Images are built using a combination of driver scripts, and functions defined in `segmentation.py`. The goal of these images is to illustrate where the models are identifying used and not used sections of the wood samples.

#### Window Function

`segmentation.py` defines a `window` function, which will use several parameters to methodically slice up an image and apply a provided model to the slices. The end result is a matrix containing the resulting prediction values. Further details regarding the functions parameters follow:

*  an original image (`original_img`) in BGR order (presumably loaded by opencv, but not necessarily)
*  a function (`model_fnc`) for performing any necessary preprocessing on the images sliced out of the original image, and then applying the model in question. The function should return a number between 0 and 1 representing the probability that the image is used.
* the window size (`window_size`) to use in slicing the image. the size is represented as a tuple, where the first number refers to the number of rows, and the second the number of columns for the slice.
* the step size in pixels for rows (`row_step`) which defines how much the windowing loop will move for each row step. 
* the step size in pixels for columns (`col_step`) which defines how much the windowing loop will move for each column step

At it's core the function uses a double for loop to iterate through the rows, and then columns, creating cropping or slices of the original image `row_step`, and `col_step` away from each other. The first couple sliced images are listed below as an example:

Result Matrix Row Index | Result Matrix Col Index | upper left row | upper left col
----------------------|-----------------------|----------------|----------------
0 | 0 | 0 | 0
0 | 1 | 0 | `col_step`
0 | 2 | 0 | `2*col_step`
1 | 0 |  `row_step` | 0
1 | 1 | `row_step` | `col_step`
1 | 2 | `row_step` | `2*col_step`
2 | 0 | `2*row_step` | 0
2 | 1 | `2*row_step` | `col_step`
2 | 2 | `2*row_step` | `2*col_step`

More generally the boundary of the original image slice can be related to the index of the resulting matrix:
upper_left_row = `matrix_row*row_step`
upper_left_col = `matrix_col*col_step`
lower_right_row = `matrix_row*row_step + window_size[0]`
lower_right_col = `matrix_col*col_step + window_size[1]`

#### Heatmap Function

`segmentation.py` defines a `heatmap` function, which is responsible for using the matrix generated by the [Window Function](#Window-Function), and the original image, to generate a heatmap of the model classification results overlaid on the original image using a specified alpha value (which defines the level of transparency).

The heatmap function takes the following parameters: 

 * an original image (`original_img`) on which the heatmap generated from the matrix will be overlaid
 * the result matrix (`matrix`) presumably (but not necessarily) generated by the [Window Function](#Window-Function)
 * (__optionally__) the `alpha` value, which defines how transparent the overlaid image will be. Defaults to 0.4

 Heatmap is implementing with heavy reliance on the cv2 package which wraps the image processing library openCV. We using `cv2.resize` to expand the matrix so that it matches the size of the image (by default openCV will use linear interpolation on the image to do this resizing. see [docs](https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html#Scaling) for more details). 

 The function then scales the values from 0 - 1 (the assumed bounds of the provided matrix) to 0 - 255 (as unsigned 8-bit integers). Next it applies a colormap, to generate a full size bgr image we can use for the final output. The colormap is applied using `cv2.applyColorMap` and the `cv2.COLORMAP_JET` option. See [docs](https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65) for more details. Blue areas are classified as Not Used, Red areas are classified as Used, and Green (as the middle), represents values close to 50%, implying the model cannot distinguish them one way or the other.


### Blur Images

The images output to this directory are actually plots that show the accuracy of the test set vs. the amount of bluring applied to the test data. The goal of these tools is identify how resilient the models are to additional blur within the sample images.

Note: This section may change further to accommodate a more common approach that can be applied to both the neural network, and classical machine learning approaches. TODO: Remove this note for final submission

#### Get Blurs Function

The file `utils.py` defines the function `get_blurs_acc` which is responsible for iterating on a provided set of blur size values (the number of pixels in length for the window the blurring operation applies), to apply that blurring to all test data, and then use the specified model to predict the classification for the full test set. We then aggregate that data to produce a plot showing the relationship between the amount of blur being applied, and classification capacity of the model.

NOTE/TBD: In reviewing this code, I noticed that there is logical error behind this process. I had applied the blurs to the already preprocessed images (which are 224x224 to match the necessary inputs for the VGG model). We'll need to adjust the algorithm to operate on the full cropped image data, and then apply image preprocessing again for each image, each time the blur iterates. This will lead to increased runtime (likely significantly), but also likely explains why the response of the VGG network seemed so overly-sensitive to the blur window size.

