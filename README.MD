# Problem & Results 

## What I'm Modeling
The goal of this code is to create a circle detector that can find the location (x, y, radius) of a circle in an image with arbitrary noise. 

## Test Statistic
For my test statistc, I used the "intesection over union" (IOU) because it's easy to understand.

I looked at the % of test data samples with IOU over 10%, 25%, 50%, 75%, 90%, and 95% because I find it more valuable to visualize the distribution than to pick a threshold.

## Results 
I tested the model on a 10,000 image test dataset and achieved the following performance: 
MSE Loss: 5.88713338428412
% IOUs over threshold:
>10%: 99.8%,
>25%: 99.6%
>50%: 97.8%
>75%: 83.5%
>90%: 35.5%
>95%: 7.6%

# How the Code Works 

## Requirements
I used Pytorch to define and train the model and Optuna to train the hyperparameters.

You'll need pytorch, numpy, optuna, and python 3.11. You'll need matplotlib and scikit-learn if you want to make plots of the circles and jupyter notebook if you want to see my data exploration.

## Running the Code
The model input is a tensor of noisy, 100x100 images that each contain circles. Its output is a tensor representing [row, column, radius].

Before running a model, set the model and training configs in model_config.py.

To run the model, use run_and_validate_model.py to run, train, and test models. You can set load_model = True to load a trained model (specify the path in the script) or set load_model = False to create a new model.

## Other Files
- cnn.py defines the CNN class. Each model is a CNN instance
- study.py sets up Optuna studies to tune hyperparameters. Once I tuned them, I updated the values in the model.config file 
- model_trainer.py defines the ModelTrainer class, which contains all the functions to train, validate, and save the CNN
- circle_dataset.py defines the CircleDataset class, which lets us easily construct training and validation datasets in tensor form
- circle_detection.py contains the provided helper functions. I added a function to help count number of samples with IOUs over certain thresholds
- data_exploration.ipynb has some scratchwork I did when evaluating the MSE as a loss function and looking through some data examples to make sure I wasn't missing any key information

# Model Architecture & Training Considerations

## Model Architecture
The model is a convolutional neural net and is made up of two types of blocks that are repeated:
- Convolution block: convolution layer -> batchnorm -> ReLU
- Dense block: fully connected layer -> batchnorm -> ReLU

The model's flow is:
- Convolution block x 3
- Dropout 
- Convolution block with dilation
- Convolution block with stride = 2
- Dropout 
- Dense block x 2
- Fully connected layer that predicts a 1D tensor with 3 elements for [row, column, radius]

## Architecture Considerations
- I start with several convolutional layers and end with a few fully connected layers to enable the model to learn image features and then translate that into the row, column, radius form we need
- I added a convolutional layer with dilation to help the model with feature abstraction
- I added the convolutional layer with stride = 2 to compress the number of parameters before the first fully connected layer
- Included dropout layers to help the model generalize/not overfit
- Included batchnorm for the same reason and to speed up training

I originally had used max pooling to compress down the spatial representation as I increased the channels out of each convolution layer, but I think too much spatial information was getting discarded with early max pooling layers. I got rid of those and added a strided convolution block to allow the model to learn what information it needs to preserve as we compress some parameters (as the first dense layer accounts for a *ton* of parameters).

## Choice of Loss Function
Since the IOU isn't differentiable, we can't use it for our loss function. I couldn't find an off-the-shelf differentiable version of it for circles, so I could either 1) make one myself or 2) use a well-understood loss function that is a reasonable proxy. (Or of course choose a new test statistic, but I think the IOU has strong benefits.)

After exploring the data (see data_exploration notebook), I landed on using the MSE because it's reasonably correlated with the IOU, especially when MSE loss is low (~.7). Based on examining the charts, we should be able to get the IOU to ~90% on average. That seemed reasonable to me and worth the benefit of getting to work with a well-understood (and well behaved) loss function, but if our task needed the extra 5%+, we could make a loss function that's a better approximation of the IOU.

## Improving Performance and Training Speed
Here are some steps I took to improve the model's performance:
- Manually evaluated Adam and SGD and chose Adam for an optimizer 
- Used weight decay for more regularization 
- Used a learning rate scheduler 
- Used Optuna to tune these hyperparameters:
    - n channels that the convolutional output
    - n outputs for the first 2 fully connected layers
    - batch size
    - weight decay parameter
    - alpha and gamma
    - p(dropout) for each dropout layer (which led to eliminating the 3rd dropout layer)

## Things I would do if I had more time and if higher accuracy were important:


The fundamental issue I see with the model is it converges quickly and then fails to learn after ~the 7th epoch. I would start by trying to figure out why that is. 
- Review architecture choices for potential improvements
- Optimize hyperparameters over a greater range of choices and with more data
    - this may help select hyperparameters that do well in late epochs, where my model stops gaining much from training
- Normalize the targets
    - the radius' mean was ~60% the x and y coordinate mean - normalizing this could increase learning gains, particularly as we get to very low loss
- Train across more noise levels (rather than using the fixed stdev for noise)
- Train with cropped images 
Outside of that, I would also want to:
- QA the provided functions to make sure they're working as expected (just a good habit)
- Make a loss function that more closely correlates with IOU (if IOU is the right test statistic)