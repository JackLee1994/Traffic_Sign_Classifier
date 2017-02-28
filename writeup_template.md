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

[image1]: ./image/exploratory_visualization.png "Visualization"
[image2]: ./image/hist.png "Grayscaling"
[image3]: ./image/image1.jpg "Random Noise"
[image4]: ./image/image2.jpg "Traffic Sign 1"
[image5]: ./image/image3.jpg "Traffic Sign 2"
[image6]: ./image/image4.jpg "Traffic Sign 3"
[image7]: ./image/image5.jpg "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

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
![alt text][image1]
The code for this step is contained in the third code cell of the IPython notebook.  

Here is an hist of my X_train

![alt text][image2]

###Design and Test a Model Architecture

####1. This is my process of train
Training...

EPOCH 1 ...
Validation Accuracy = 0.749

EPOCH 2 ...
Validation Accuracy = 0.814

EPOCH 3 ...
Validation Accuracy = 0.890

EPOCH 4 ...
Validation Accuracy = 0.873

EPOCH 5 ...
Validation Accuracy = 0.868

EPOCH 6 ...
Validation Accuracy = 0.886

EPOCH 7 ...
Validation Accuracy = 0.888

EPOCH 8 ...
Validation Accuracy = 0.891

EPOCH 9 ...
Validation Accuracy = 0.908

EPOCH 10 ...
Validation Accuracy = 0.899

EPOCH 11 ...
Validation Accuracy = 0.903

EPOCH 12 ...
Validation Accuracy = 0.913

EPOCH 13 ...
Validation Accuracy = 0.918

EPOCH 14 ...
Validation Accuracy = 0.893

EPOCH 15 ...
Validation Accuracy = 0.916

EPOCH 16 ...
Validation Accuracy = 0.917

EPOCH 17 ...
Validation Accuracy = 0.880

EPOCH 18 ...
Validation Accuracy = 0.931

EPOCH 19 ...
Validation Accuracy = 0.926

EPOCH 20 ...
Validation Accuracy = 0.921

EPOCH 21 ...
Validation Accuracy = 0.911

EPOCH 22 ...
Validation Accuracy = 0.919

EPOCH 23 ...
Validation Accuracy = 0.922

EPOCH 24 ...
Validation Accuracy = 0.915

EPOCH 25 ...
Validation Accuracy = 0.922

EPOCH 26 ...
Validation Accuracy = 0.915

EPOCH 27 ...
Validation Accuracy = 0.924

EPOCH 28 ...
Validation Accuracy = 0.939

EPOCH 29 ...
Validation Accuracy = 0.920

EPOCH 30 ...
Validation Accuracy = 0.925

Model saved

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
Test Accuracy = 0.927
* validation set accuracy of ? 
* test set accuracy of ?
0.4

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

Speed limit (60km/h) 0.92727
Ahead only 0.999867
General caution 1.0
Right-of-way at the next intersection 1.0
Priority road 1.0

End of speed limit (80km/h) 0.0725797
Turn right ahead 9.79449e-05
Go straight or left 3.85787e-09
Beware of ice/snow 1.24023e-08
Right-of-way at the next intersection 4.52573e-10

Speed limit (80km/h) 9.72188e-05
Turn left ahead 3.50892e-05
Keep right 4.89764e-21
End of speed limit (80km/h) 1.17456e-09
Speed limit (50km/h) 6.61042e-12

Speed limit (50km/h) 5.2937e-05
Keep right 1.97339e-07
Turn right ahead 3.12704e-29
Road work 2.85069e-10
Stop 9.53769e-14

Children crossing 1.38645e-09
Go straight or right 1.0432e-09
Keep left 4.36975e-34
Priority road 4.95786e-16
Yield 5.06717e-18

For the second image ... 
