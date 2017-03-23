# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---
### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. 
---

[//]: # (Image References)

[trainingdata_org]: ./writeup_images/trainingdata_org.png "Example training data by class"
[trainingdata_org_histogram]: ./writeup_images/trainingdata_org_histogram.png "Original trainingdata histogram"
[affine_transformed]: ./writeup_images/affine_transformed.png "Affine transformed images"
[trainingdata_egalized_histogram]: ./writeup_images/trainingdata_egalized_histogram.png "Egalized trainingdata histogram"
[trainingdata_preprozessed]: ./writeup_images/trainingdata_preprozessed.png "Training Data preprozessed"
[5_traffic_signs]: ./writeup_images/5_traffic_signs.png "German traffic signs from the web"
[result_sign_stop]: ./writeup_images/result_sign_stop.png "Results for stop sign"
[result_sign_80]: ./writeup_images/result_sign_80.png "Results for speed limit 80 km/h"
[result_sign_100]: ./writeup_images/result_sign_100.png "Results for speed limit 100 km/h"
[result_sign_no_vehicles]: ./writeup_images/result_sign_no_vehicles.png "Results for no vehicles"
[result_sign_yield]: ./writeup_images/result_sign_yield.png "Results for yield"
[graph_sign_stop]: ./writeup_images/graph_sign_stop.png "Graph for stop sign"
[graph_sign_80]: ./writeup_images/graph_sign_80.png "Graph for speed limit 80 km/h"
[graph_sign_100]: ./writeup_images/graph_sign_100.png "Graph for speed limit 100 km/h"
[graph_sign_no_vehicles]: ./writeup_images/graph_sign_no_vehicles.png "Graph for no vehicles"
[graph_sign_yield]: ./writeup_images/graph_sign_yield.png "Graph for yield"

### Files included
*  The programmed Solution [Traffic_Sign_Classifier.ipynb](https://github.com/kaiulrich/kulrich_CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) 

* The saved result  [Traffic_Sign_Classifier.html](./Traffic_Sign_Classifier.html)

* The test images [images](https://github.com/kaiulrich/kulrich_CarND-Traffic-Sign-Classifier-Project/blob/master/images)


---

### Data Set Summary & Exploration

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

---


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

##### c. shuffel data

The taindata had been shuffeld before the splitting.
For the code look at "Split data" chapter of the ipython notebook. 

---

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

To cross validate my model, I randomly split the training data into a training set (80 %) and validation set (20%). I did this by using 'train_test_split' function of the sklearn.model_selection package. 
For the code look at "Split data" chapter of the ipython notebook. 

My original training set had 34799 images. 
My final training set had 53904 number of images. 
My final validation set and test set had 13476 and 12630 number of images.

---

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

For the code look at "Model Architecture" chapter of the ipython notebook.

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

---

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following hyperparameters


EPOCHS = 200:

BATCH_SIZE = 300

**Arguments used for the AdamOptimizer**

learning_rate = 0.001

**Argument used by dropout**

keep_prop = 0.75 

**Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer:**

mu = 0
sigma = 0.2 

For the code look at "Training Pipeline" chapter of the ipython notebook.

---

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is in the "Evaluate the Model" chapter of the ipython notebook.

My final model results were:

* Validation Accuracy = 0.976
* Test Accuracy = 0.939


I started with [LeNet-5](http://yann.lecun.com/exdb/lenet/)  Architekture for letter recognition. LeNet-5 solved a similar kind of problem: 
"LeNet-5 Network is designed to recognize visual patterns directly from pixel images with minimal preprocessing.  It can recognize patterns with extreme variability (such as handwritten characters), and with robustness to distortions and simple geometric transformations."
 
 The first test with the Network and unpreprozessed images  had a Validation and Test Accurcy around 0.7. 

The next configuration I tesed had been with a min max normalisation an the Validation encreased over 0.92 the Test Accurcy 0.83 
After adding the the trainingdata egalisation it wasn't mutch different.

Than made was to encrease the sigma ( randomly defines variables for the weights and biases for each layer) from 0.1 to 0.2. The Validation Accurcy encreased over 0.95. the Test Accurcy stayed arround 0.870

Than I added the dropout layers. I chosed to put them to the first and second Fully Connected layers. The Validation Accurcy encreased a bit 0.970. But Test Accurcy enccreased to 0.927. But the Validation Accurcy encreased a lot slower.

Than I chaned the number of full connected layer, I tested 2, 3  layers.
I got vollowing results: 

2 full connected layers
* Validation Accuracy = 0.972
* Test Accuracy = 0.921

3 full connected layers
* Validation Accuracy = 0.970
* Test Accuracy = 0.927


I decided to stay with the 3 full connected layers. because the cost are moderate and test and Test Accuracy was better.


Now I started to encrease the number of epochs.

50 epochs
* Validation Accurcy 0.970
* Test Accurcy 0.927

100 epochs 
* Validation Accurcy 0.977
* Test Accurcy 0.919

200 epochs 
* Validation Accurcy 0.976
* Test Accurcy 0.939

300 epochs 
* Validation Accurcy 0.977
* Test Accurcy 0.931


At 300 epochs the Validation Accurcy still increases  but the Test Accurcy decreases. This may indicate an offerfitting.
I decided to stay with the 200 epoch training cycle.
 ---

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][5_traffic_signs] 

I was sure that the stop and the yield image were easy. They have a unic shape. 
The speed limit 80 image is more difficult, because the 20 30 60  signs are very similar. 
I was sure the No vehicles sign is dificult to detect, because it has an white square sign below.

---

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80% (0.800). This compares favorably to the accuracy on the test set of 0.939.
For the code look at "Analyze Performance" 

---

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the code for making predictions on my final model look Ipython notebook chapter "Output Top 5 Softmax Probabilities For Each Image Found on the "

Here are the results of the prediction:

**sign to detect : Stop (14)**

* 14 - Stop                           - 1.0000000000
* 29 - Bicycles crossing              - 0.0000000000
* 17 - No entry                       - 0.0000000000
![alt text][result_sign_stop]
![alt text][graph_sign_stop]


**sign to detect : Speed limit (80km/h) (5)**

 * 5 - Speed limit (80km/h)           - 1.0000000000
 * 3 - Speed limit (60km/h)           - 0.0000000075
 * 2 - Speed limit (50km/h)           - 0.0000000000
 ![alt text][result_sign_80]
  ![alt text][graph_sign_80]

**sign to detect : Speed limit (100km/h) (7)**

 * 7 - Speed limit (100km/h)          - 1.0000000000
 * 5 - Speed limit (80km/h)           - 0.0000000000
 * 2 - Speed limit (50km/h)           - 0.0000000000

 ![alt text][result_sign_100]
 ![alt text][graph_sign_100]

**sign to detect : No vehicles (15)**

* 9 - No passing                     - 1.0000000000
* 13 - Yield                          - 0.0000000077
* 10 - No passing for vehicles over 3.5 metric tons - 0.0000000000

 ![alt text][result_sign_no_vehicles]
 ![alt text][graph_sign_no_vehicles]
 
**sign to detect : Yield (13)**

* 13 - Yield                          - 1.0000000000
* 42 - End of no passing by vehicles over 3.5 metric tons - 0.0000000000
* 38 - Keep right                     - 0.0000000000

 ![alt text][result_sign_yield]
 ![alt text][graph_sign_yield]
 
 
 * Like expected there was no problem to indicate the "Stop" and "Yield" sign.
 * The "Speed limit (80km/h)" was indicated right as well. The detection hat no problem to indicate two characters and the "2 closed circle" number 8. With a big distance It detects the "1 closed circle - 1 open circle" number 6 and the "2 open circle" number three  as a possible result. 
 * The "No vehicles" sign wasn't detected. The detection got the red cirkle of the "No passing" sign und got a horizontal strucktur. 

---


## Reflection 


### Pre-Prozessing

Preprozessing the images had a very strong influence on the detection result.
min max normalisation did a good job.
After analysing the images I recognized the the shape of the signs are verry clear strucktured. So the shape could by a strong characteristic of the sign and should come to the fore.  To work with grayscaled images with gaussian blur filtering and Canny edge detection  could be an intersting approach. 

### Egalize number of images by class 

The training set has a strong influence on the detection result.
I my case the effect wasn'd to big. The reason could be, that most of the testet images were in a class where the original training set had a relativly high number of pictures. (Class 5 Speed limit (80km/h), Class 7 Speed limit (100km/h)) or the shape was easy to detect (Class 14 Stop) 
The my be statisticaly more efficient ways to fill up the week classes, but it would by better to encrease the number of original pictures.

### Training epochs and hyperparameters

The results for same configuration differ. The reason are the random start weights and biases. It is worth to run the training a couple of times and take the best result.

It takes a lot of time to train the network. So I had to make a compromize between batchsize, epochs, architectur complexity and training set size to find the best architekture. It is importand to keep in mind that in this sence the trend of the Validation and Test Accuracy are not linear.

### New image selection

I did not expect the quality of the detection so far. To get more information about the quality of the detection it would be good to choose more dificult examples. 

