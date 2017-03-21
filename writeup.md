#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---

[//]: # (Image References)

[trainingdata_org]: ./writeup_images/trainingdata_org.png "Example training data by class"
[trainingdata_org_histogram]: ./writeup_images/trainingdata_org_histogram.png "Original trainingdata histogram"
[affine_transformed]: ./writeup_images/affine_transformed.png "Affine transformed images"
[trainingdata_egalized_histogram]: ./writeup_images/trainingdata_egalized_histogram.png "Egalized trainingdata histogram"

[trainingdata_preprozessed]: ./writeup_images/trainingdata_preprozessed.png "Training Data preprozessed"

[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


### Files included
*  The programmed Solution [Traffic_Sign_Classifier.ipynb](https://github.com/kaiulrich/kulrich_CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

* The saved result  [Traffic_Sign_Classifier.html](./Traffic_Sign_Classifier.html)

###Data Set Summary & Exploration

#### 1.  Basic data summary

for the code look at "Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas" chapter of the ipython notebook. 

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.


Here is an exploratory visualization of the data set. 

#### Example training data by class

for the code look at "Trainingdata images before preprozessing" chapter of the ipython notebook. 
![alt text][trainingdata_org]

#### Histogram of training data
for the code look at "Histogram of training data" chapter of the ipython notebook. 
![alt text][trainingdata_org_histogram]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

##### a. Egalize number of images by class
Looking at the test data, I noticed that the number of images by class are not uniformly distributed.
There some very strong classes like Class 1 or 2 and some very week represented images like Class 0 or 6. 
In the first step I egalized the numbers of images by class and filled the classes up  with random affine transform (rotated, shifted and sheared) images of this class. 

for the code look at "Egalize number of images by classes" chapter of the ipython notebook. 

![alt text][affine_transformed]

At the end each class has a minimum of 1500 images by class.

![alt text][trainingdata_egalized_histogram]


##### b. Image normalisation

Than I chosed a min max normalisation to pre-process the data. 
I wanted to keep the 3 color layers and push the contrasts.
for the code look at "Pre-process the Data Set" chapter of the ipython notebook. 

![alt text][trainingdata_preprozessed]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

To cross validate my model, I randomly split the training data into a training set (80 %) and validation set (20%). I did this by using 'train_test_split' function of the sklearn.model_selection package. 
for the code look at "Split data" chapter of the ipython notebook. 

My original training set had 34799 images. 
My final training set had 53904 number of images. 
My final validation set and test set had 13476 and 12630 number of images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

for the code look at "Model Architecture" chapter of the ipython notebook.

My final model consisted of the following layers:

| Layer         			|     Description	        					| 
|:------------------------------:|:-------------------------------------------------------------:| 
| Input         		 	| 32x32x3 RGB image   						| 
| Convolution 5x5x3x6  	| 1x1 stride, valid  padding, outputs 28x28x6 	|
| RELU				|										|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6			        |
| Convolution  5x5x3x6 	| 5x5x16, valid  padding, outputs 10x10x16     	|
| RELU				|										|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16					|
| Flatten		           	| outputs 400        							|
| Fully connected		| outputs 120        						    	|
| RELU				|										|
| Dropout				| keep_prob = 0.75							|
| Fully connected		| outputs 84        						    	|
| RELU				|										|
| Dropout				| keep_prob = 0.75							|
| Fully connected		| outputs 43        						    	|
| RELU				|										|
| Fully connected		| outputs 43        						       	|


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following hyperparameters


EPOCHS = 50:

BATCH_SIZE = 300

**Arguments used for the AdamOptimizer**
learning_rate = 0.001

**Argument used by dropout **
keep_prop = 0.75 

**Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer:**
mu = 0
sigma = 0.2 

The the code look at "Training Pipeline" chapter of the ipython notebook.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



## TODO reflection 
- prprozessing
- selection images to test
- layers
- hüperparameters