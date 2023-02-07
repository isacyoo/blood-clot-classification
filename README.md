# blood-clot-classification

This project is a deep learning computer vision project with a medical image dataset, more specifically a set of images of Stroke Blood Clot Origin. You can find out more details about this project at https://www.kaggle.com/competitions/mayo-clinic-strip-ai.


## Preprocessing

The huge original pathology image files were not suitable to be directly used for training because of multiple reasons. The biggest issue is data I/O, as reading a huge tiff file typically takes more than 10 seconds. Another critical issue is that the input image needs to be downsampled due to computational resources. Although most CNN allows variable input sizes, it cannot take images that the computational resources, such as GPU, cannot handle. As we resize the high-resolution image to a smaller image such as 256 * 256, we will lose most of the important information and we will not be able to train our model properly. I have tackled this problem by using a Python package "OpenSlide", which helps the user to break down a huge tiff file into a set of tiles and select the ones that have relatively low mean pixel values and relatively high standard deviation of pixel values to avoid blank tiles being included. This solves both of the two problems stated above and also acts like data augmentation as the user can obtain hundreds of detailed images from a single tiff file. This also helps data scarcity faced by most of medical computer vision problems as the number of distinct images increases by a factor of hundreds.

## Initial approach

I was inspired by Sellergren et al. (2022) (https://doi.org/10.1148/radiol.212482), which was on domain-specific supervised pre-training with metric learning. BloodClotBackbone class in model.py was implemented using this idea. However, I discarded this and moved to a typical CNN backbone with a classification head as I did not see significant increase in model accuracy. t_sne.py has util functions that visualises how well embeddings produced by a backbone are clustered according to their labels.

## Training

I am familiar with both PyTorch and PyTorch-Lightning, which is a high-level wrapper of PyTorch that enables faster development and reduces lines of code, and is fully compatible with vanilla PyTorch. Three hyperparameters related to ResNet architecture had to be determined through multiple studies. For each ResNet configuration, several nuisance hyperparameters were tuned using a Python package "optuna". Refer to tuning.py and exp.py.
