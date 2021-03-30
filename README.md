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


### Segmentation Images Directory

This directory is intended to contain resulting data from the segmentation evaluation process.

My initial attempt at segmenting the data based on the code in the repo can be downloaded here:

* https://gwu.box.com/s/22rik4vzfj279pre9qhud9f5p1l6yfps

### Blur Images Directory

This directory is intended to contain resulting data from the blur evaluation process.

My initial attempt at generating these plots based on the neural neteworks (only contains vgg at this time):

* https://gwu.box.com/s/jgjpthredw7qp0br6in2oovk8rmgaed2


### Update config.py

The `config.py` data is the sole source of directory locations, and other data inputs used by other scripts within the repository. 


## Code

This section contains brief descriptions of the scripts found in the `src/` directory of the repository.

