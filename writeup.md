# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataset]: ./dataset.png "dataset"
[distribution]: ./class_distribution.png "distribution"
[transformations]: ./transformations.png "transformations"
[augmentation]: ./augmentations.png "augmentations"
[websign1]: ./webimages/sign1.jpg "Traffic Sign 1"
[websign2]: ./webimages/sign2.jpg "Traffic Sign 2"
[websign3]: ./webimages/sign3.jpg "Traffic Sign 3"
[websign4]: ./webimages/sign4.jpg "Traffic Sign 4"
[websign5]: ./webimages/sign5.jpg "Traffic Sign 5"
[websign6]: ./webimages/sign6.jpg "Traffic Sign 6"
[websign7]: ./webimages/sign7.jpg "Traffic Sign 7"
[websign8]: ./webimages/sign8.jpg "Traffic Sign 8"
[websign9]: ./webimages/sign9.jpg "Traffic Sign 9"
[websign10]: ./webimages/sign10.jpg "Traffic Sign 10"
[websign11]: ./webimages/sign11.jpg "Traffic Sign 11"
[websign12]: ./webimages/sign12.jpg "Traffic Sign 12"
[websign13]: ./webimages/sign13.jpg "Traffic Sign 13"
[websign14]: ./webimages/sign14.jpg "Traffic Sign 14"
[webpred1]: ./webpredictions1.PNG "Web Predictions 1"
[webpred2]: ./webpredictions2.PNG "Web Predictions 2"
[webpred3]: ./webpredictions3.PNG "Web Predictions 3"
[visualization1]: ./visualization1.PNG "Visualization 1"
[visualization2]: ./visualization2.PNG "Visualization 2"
[visualization3]: ./visualization3.PNG "Visualization 3"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. Jupyter Notebook with the source code and visualization is available at [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the Pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

The following figure shows 5 random images from each category in training, validation and test sets:

![alt text][dataset]

I observed that the images were created under various illumination conditions. Some images are very dark and some images are very bright. Also the size and viewing angle of the same traffic sign varies across the images, e.g. see the images of "Yield" sign. Later on, in the data preprocessing step, I will take care of these differences in illumination, traffic sign position and size.

The following figure shows the fraction of each category in the training, validation and test sets:

![alt text][distribution]

According to this figure, this is unbalanced dataset. The most common traffic sign is "Speed limit (50 km/h)" and it amounts for 6% of the training images. The least common traffic sign in "Speed limit (20 km/h)" and it amounts for 0.5% of the training images. This graph also show that training and test sets (red and blue bars respectively) have a similar distribution of categories, whereas the validation set (green bars) has a different distribution of labels. Still the differences in the distribution of labels  in the training and validation sets are not dramatic and I ignore them in the subsequent training pipeline.

### Design and Test a Model Architecture

#### 1. Data Preprocessing and augmentation

I showed in the previous section that images were taken under different illumination condition. My first preprocessing step is to normalize the images so that they will be bright and sharp. I followed the *contrast normalization pipeline* of [1]. This pipeline has 4 steps:

1. **Image Adjustment**. In this step I increased image contrast by mapping pixel intensities to new values such that 1% of the pixels have low and high intensities.
2. **Histogram Equalization**. In this step I further increased image contrast by transforming pixel intensities such that the output image histogram is close to uniform.
3. **Adaptive Histogram Equalization** operates on tiles rather than the entire image: the image is tiled in non-overlapping regions of 6x6 pixels each. Every tile’s contrast is enhanced such that its histogram becomes roughly uniform. Since the images are 32x32, and the height and width are not multiples of 6, this transformation pads image with zeros to get 36x36 image.
4. **Contrast normalization**. In this transformation I increased edge visibility through ﬁltering the input image by a diﬀerence of Gaussians. I subtracted the image blurred with 3x3 Gaussian filter from the image blurred with 5x5 Gaussian filter.

Note that the first three transformations are not performed in the original RGB-color space but rather in YUV space. I applied transformations 1-3 to Y channel, which is a gray scale version of the original image. After transforming Y channel I converted an image back to RGB space.

Here is an example of several images of "Dangerous Curve" traffic signs as they go through the contrast normalization pipeline:

![alt text][transformations]

Notice that although original images have different brightness and in some images it is very hard to see a black curve inside the sign. After the contrast normalization pipeline the black curve is clearly seen in all images.

I ran all training, validation and test images through contrast normalization pipeline. I didn't include mean normalization of the image since we didn't observe that it improves validation error.

We decided to generate additional data to be able to train larger neural network and to have the resulting neural network robust to the noisy images.

To add more data to the training set, we applied the following four transformations to each image in the training set after contrast normalization pipeline:

1. **Scale**. This transformation creates images with traffic signs very close and very far. We scale an image horizontally and vertically. A scaling factor in each direction is drawn uniformly from [0.9,1.1]. After the scaling we take the central 32x32 patch of the image.
2. **Rotate**. This transformation creates images where traffic signs are slightly rotated. We rotate the original image by angle drawn uniformly from the range of [-5,5] degrees.
3. **Translate**. This transformation creates images where the traffic sign is not exactly in the center. We shift the original image horizontally and vertically. The shift magnitude in each direction is drawn uniformly from (-3,-2,-1,1,2,3) pixels.
4. **Shear**. This transformation rotates traffic around its vertical axis. This create an effect that the traffic sign was not in front of the camera, but to the side of it.

Here is an example of original images and transformed ones:

![alt text][augmentation]

The augmented training set has the images after contrast normalization pipeline and after each of the above 4 transformations. Altogether, the augmented training set has 173995 images.


#### 2. Model Architecture

My final model has the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:-----------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1x1   | outputs array 32x32x12                   |
| ReLU              |                                   |
| Convolution 5x5   | 1x1 stride, valid padding, outputs array 28x28x24 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs array 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs array 10x10x64  |
| ReLU         |                                     |
| Max pooling	      	| 2x2 stride,  outputs array 5x5x64 				|
| Flattening          | outputs a vector of length 1600
| Fully connected		| outputs a vector of length 120        									|
| ReLU              |                                     |
| Fully connected   | outputs a vector of length 84                          |
| ReLU              |                                     |
| Fully connected   | outputs outputs a vector of length 43                          |
| Softmax				| outputs 43 probabilities       									|

#### 3. Training the model

To train the model, I used Adam optimizer with batch size 128 and learning rate 0.001. My objective function was cross-entropy + regularization_rate * L2-regularization of all weights and biases in the network. I used regularization rate = 0.001. In the beginning of each epoch I shuffled the training set. I did optimization for at most 200 epochs and user early stopping technique. After each epoch I measured the accuracy of the model over the validation set. I stopped the training if there was no improvement in validation accuracy over 20 iterations.

During the training process I saved a model after each epoch. At the end of the training I chose my final model as the one that has the best validation accuracy.

#### 4. Describe the approach taken for finding a solution.

My final model results were:
* training set accuracy of 99.374
* validation set accuracy of 98.413
* test set accuracy of 96.302

I started with LeNet-5 architecture, since this is a relatively small network that works well in image classification tasks. I've got training accuracy 99.253% and validation accuracy 91.179% with LeNet-5. This is a clear overfitting to the training set. Then I tried to adjust images so that images in validation set look more similar to the images in the training set. The normalization with mean and standard deviation resulted in validation accuracy 92.517. Then I replaced this normalization with the contrast normalization pipeline and got validation accuracy 94.717. In both cases the training accuracy was more than 99%, which means that there was still overfitting to the training set. One of the reasons for this overfitting is that original training set does not have enough images to train a robust LeNet-5 network. In the next step I increased the training set by adding images from 4 transformations. This indeed reduced overfitting, raised validation accuracy to 96.032, but lowered the training accuracy to 0.98545. To reduce overfitting further, I added to the objective function L2 regularization of all weights and biases. This raises validation accuracy to 96.372, while lowering the training further to 97.659.

At this point the difference between training and validation accuracy was less than 2%, which means that overfitting was not significant. However, since the training accuracy dropped from 100% to 97%, the network started to underfit. In the next experiments I changed the architecture of the network to make it more powerful and to reduce underfitting. I added 3 1x1 convolution filter before the first layer. This layer tries to find a representation of the image that is better than the original RGB representation. After this step I've got validation accuracy 96.712. Then I doubled the number of feature maps in the convolutional layers to (6,12,32) and increased the validation accuracy to 97.778. I double the number of feature maps further to (12,24,64) and got the validation accuracy of 98.413. I noticed that increasing the number of feature maps further does not improve validation accuracy. I also tried to tune regularization rate and learning rate. I've found that the best value for both of them is 0.001.

The following table summarizes the iterations described above:

| Data set      		|     Objective function | Network Architecture | Train Accuracy	        					|    Validation Accuracy  |
|:---------------------:|:-----------------------------:|:--------------------:|:---------:|:-----------|
| Original data        		|  Cross-entropy  							| LeNet-5         |   99.253     | 91.179      |
| Mean normalization     |  Cross-entropy               | LeNet-5           |   100     |   92.517   |
| Contrast normalization  |  Cross-entropy    | LeNet-5           | 99.89   |   94.717    |
| Contrast normalization, data augmentation   |  Cross-entropy      |  LeNet-5      |  98.545 |  96.032  |       
| Contrast normalization, data augmentation   |  Cross-entropy, L2 regularization   |  LeNet-5 |  97.659  | 96.372  |
| Contrast normalization, data augmentation   | Cross-entropy, L2 regularization |  3 1x1 convolutions, LeNet-5   |   98.557     |   96.712        |
| Contrast normalization, data augmentation   | Cross-entropy, L2 regularization | 6 1x1 convolutions, LeNet-5 with 12 and 32 convolutions in the first two layers  |  98.795      |   97.778        |
| Contrast normalization, data augmentation   | Cross-entropy, L2 regularization | 12 1x1 convolutions, LeNet-5 with 24 and 64 convolutions in the first two layers | 99.374   | 98.413 |        

The final architecture has training accuracy over 99%. This means that there is no underfitting. Also, since the difference between training and validation accuracies is around 1%, I assumed the overfitting is not significant.
At this point I measured test set accuracy and got 96%. This an evidence that the network still overfits to training and validation sets. Despite that the network is still powerful, since it misclassified only  (1-0.96302) * 12650 = 468 out of 12650 test images.   

### Test a Model on New Images

Here are images of fourteen German traffic signs that I took from [Driving Berlin video](https://www.youtube.com/watch?v=Givfsvsc948):

![alt text][websign1] ![alt text][websign2] ![alt text][websign3]
![alt text][websign4] ![alt text][websign5]![alt text][websign6] ![alt text][websign7] ![alt text][websign8]
![alt text][websign9] ![alt text][websign10]![alt text][websign11] ![alt text][websign12] ![alt text][websign13]
![alt text][websign14]

This video is taken from the central area of Berlin, Germany and thus has traffic signs that my network was trained on. Moreover, this video was taken from moving car and parts of the video were taken while there was rain. This setup simulates the possible use case of deploying my model in a car.

Original images have different sizes, but they were scaled to 32x32 before scoring. All images are blurry, especially images #5-#7, and hence might be difficult to classify. Also image #4 (roadwork) has white patches on top of the red boundary. This might be a significant obstacle for my model.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General caution     		| General caution 									|
| Priority road    			| Priority road										|
| Bumpy road 					| Bumpy road										|
| Road work	      		| Road work				 				|
| Keep right			| Keep right     							|
| Speed limit (30 km/h)			| Speed limit (30 km/h)    							|
| Bicycles crossing         | Bicycles crossing                     |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Turn right ahead    | Turn right ahead   |
| Yield               |  Yield             |
| Go straight or left | Go straight or left |
| Speed limit (50 km/h) | Speed limit (50 km/h) |
| Ahead only | Ahead only |
| Traffic signals | Traffic signals |

The model was able to correctly guess 14 of the 14 traffic signs, which gives an accuracy of 100%. Recall that the test set accuracy is 96.3%, measured over 12K images. 14 images is a very small sample to decide if performance over web images is better than over the test images. But nevertheless 100% accuracy over 14 images confirms that our model is very accurate over German traffic signs.   

The code for making predictions on my final model is located in the 22nd cell of Jupyter notebook. For all images the model generates very confident predictions, with top-1 probability at least 99%. The following image shows top-5 predictions for each image, along with corresponding predicted probabilities.

![alt text][webpred1]
![alt text][webpred2]
![alt text][webpred3]

### (Optional) Visualizing the Neural Network (See Step 4 of the Jupyter notebook for more details)

I visualized the values of 24 feature maps that are created by the second convolutional layer. The following figures show 3 original images, the images created by normalization pipeline (see Section ) and the resulting feature maps.

#### General caution sign
![alt text][visualization1]
I observed that many feature maps (e.g. 0, 1, 3-6, 12-14, 16-18, 22, 23) capture one side of the red triangle. Also several feature maps (0,1,10,12,14,18,22,23) capture the exclamation sign.

#### Priority road sign
![alt text][visualization2]
Priority road sign has 4 pairs of parallel edges. Each such pair has transition from yellow to white and then from white to gray. Three pairs  of edges are activated in feature maps 0,3,4. I don't see a strong activation of the top right pair of edges. I also observed a strong activation of individual edges in feature maps 1,7,9,10,12-15,21.

#### Bumpy road sign
![alt text][visualization3]
Bumpy road sign is very challenging since it has small details. Several feature maps (0,9,12,16-18,22) captured one or two sides of the read triangle. Feature maps 5 and 6 capture the bumpy road inside the triangle.

### References
[1] D. Ciresan, U. Meier, J. Masci and J. Schmidhuber. Multi-Column Deep Neural Network for Traffic Sign Classification. Neural Networks 32: 333-338, 2012.
