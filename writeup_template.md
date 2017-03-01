#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/hist.png "hist1"
[image2]: ./image/exploratory_visualization.png "Grayscaling"
[image3]: ./image/hist2 "hist2"
[image4]: ./image/image1.jpg "Traffic Sign 1"
[image5]: ./image/image2 "Traffic Sign 2"
[image6]: ./image/image3 "Traffic Sign 3"
[image7]: ./image/image4 "Traffic Sign 4"
[image8]: ./image/image5 "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JackLee1994/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
Number of training examples = 34799
* The size of test set is ?
Number of testing examples = 12630
* The shape of a traffic sign image is ?
Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set is ?
Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I didn't convert the images to grayscale because I thought that RGB image include more imformation.

Here is an example of all traffic sign image.

![alt text][image2]

As a last step, I normalized the image data because he datatasets consist of 32x32 rgb images. I normalized the samples to the range of -0.5 and 0.5.Normalization is important in CNNs because of their learning process. CNNs learn by continually adding gradient error vectors computed from backpropagation to various weight matrices If I dont scale the input training vectors the ranges of our distributions of feature values would be different for each feature,resulting in corrections in each differing dimension.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by splitting the source data

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because data set is small. To add more data to the the data set, I used the following techniques because the images that we get are different angle. I changed the angle of image to get more input data

Here is processed data.

![alt text][image3]

The difference between the original data set and the augmented data set is the following 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

My final network architecture takes batches of 32x32 greyscale images as input and sends those through 3 5x5 convolutional layers with depths of 32,64, and 128, followed by a fully connected layer with a width of 512. Each of the convolutional layers is followed by a tanh activation function then a 2x2 max pooling layer. The output layer consisits of a 43-class classifier.
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I tried changing with learning rate, epochs, and batch size many times. In the end, I picked: learning rate = 0.001 epochs = 30 batch size = 128.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of ? 
Validation Accuracy = 0.925
* test set accuracy of ?
Test Accuracy = 0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture is Lenet, because I want to test the accuracy.
* What were some problems with the initial architecture?
Accuracy is not very high. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I normalized the data set.
* Which parameters were tuned? How were they adjusted and why?
I changed some epoch to get higher accuracy.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Maybe I could add more layer, but time of training is too long.
If a well known architecture was chosen:
* What architecture was chosen?
Lenet
* Why did you believe it would be relevant to the traffic sign application?
I think that it is good to classify image
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 Test Accuracy = 0.927. It's not very bad.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)  		| Speed limit (60km/h) 									| 
| Stop   			| Ahead only 										|
| General caution					| General caution											|
| Road work	  		| Right-of-way at the next intersection					 				|
| Turn left ahead			| Priority road    							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares are not favorably to the accuracy on the test set of 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.92727       			| Speed limit (60km/h)  									| 
| 0.0725797    				|End of speed limit (80km/h) 										|
| 9.72188e-05					| Speed limit (80km/h)											|
| 5.2937e-05	      			| Speed limit (50km/h)					 				|
|1.38645e-09				    | Children crossing      							|
