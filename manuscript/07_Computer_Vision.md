

# Computer Vision

*Computer vision (CV)* is a wide field of AI. It is all about processing images: still images and moving images (videos) / analyzing and generating. Relevant questions are:

- How to retrieve and classify images and videos, e.g., which photos depict certain people?
- What is depicted in an image / which situation takes place in a video, e.g., is there a danger of a vehicle collision?
- How to generate images / videos from an internal representation, e.g., in a computer game?

In the following section, I will briefly introduce prominent computer vision applications.



## Computer Vision Applications


### Optical Character Recognition (OCR)

Optical Character Recognition (OCR) is the extraction of text from images of typed, handwritten or printed text. 

See Fig. 7.1. for the reconstruction of the text from the Wikipedia article on Michelangelo from a screenshot using [FreeOCR](http://www.freeocr.net).

![Fig. 7.1: OCR Example](images/OCR.png)


Use cases include the following ones. 

- Scanning addresses on letters
- Automatic number plate recognition
- Evaluating manually filled-out forms, e.g. in polls or surveys
- Data entry for business information systems, e.g. from invoices, bank statements and receipts etc.
- Checking documents, e.g. passports, drivers licenses, etc. 
- Archiving and retrieving text documents for which only scans are available, e.g. Google Books
- Converting handwriting in real-time (pen computing)




### Face Recognition

Face recognition is the detection of human faces in images or videos, as well as the identification of the respective persons. 

See Fig. 7.2.

![Fig. 7.2: Face Recognition](images/Face_Recognition.png)

Use cases include the following ones. 

- Identifying friends and family members in a private photo collection
- Face detection in cameras to focus and measure exposure on the face of the person being photographed
- Biometrical identification of people, e.g., for airport security
- Retrieving photos of prominent people from the Web





### Image Processing

Image processing is the wide area of post-processing digital images, particularly photos, for several reasons. Examples are:

- Changing color, brightness, contrast, smoothing, noise reduction, etc.
- Retouching, e.g., removing red eyes
- Slicing, i.e., dividing an image into different sections to be used individually
- Image restoration; see Fig. 7.3.


![Fig. 7.3: Image restoration](images/Image_Restoration.png)



### Medical Applications

Computer vision has numerous uses cases in medicine. Examples are:

- Generating images from examination procedures like [PET](Positron_Emission_Tomography)/[CT](Computed_Tomography), [MRI](Magnetic_Resonance_Imaging), ultrasound images (2D and 3D) for inspection by doctors
- Detecting anomalies in PET/CT, MRI, ultrasound images etc. automatically.

See Fig. 7.4.

![Fig. 7.4: Medical computer vision applications](images/Medical_Applications.png)




### Industrial and Agricultural Applications

Computer vision is increasingly used for automating processes in industry and agriculture. Examples are:

- Quality management in manufacturing processes
- Robot control in manufacturing processes
- Sorting fruits in agriculture

See Fig. 7.5.

![Fig. 7.5: Industrial computer vision application](images/Industrial_Application.png)


### Automotive Applications

Computer vision applications are, today, state-of-the-art in modern cars. Examples are:

- Parking aid and automatic parking
- Collision warning
- Road sign detection
- Autonomous driving

See Fig. 7.6.


![Fig. 7.6: Automotive computer vision application](images/Automotive_Application.png)



### Military, Aviation and Aerospace Applications

Also in military and aviation and aerospace industries, computer vision is applied. Examples are:

- Collision detection
- Drone and missile navigation
- Detection of enemy soldiers or vehicles
- Autonomous vehicles (see Fig. 7.7)


![Fig. 7.7: Autonomous space vehicle](images/Aerospace_Application.png)





### Computer Games and Cinema

Computer games are mostly visual. Also in modern cinema, computer vision is heavily used for special effects.
Examples are:

- Generation of moving images in computer games  
- Generation of images and scenes in animated movies  
- Motion capturing for digitizing videos taken from human actors; see Fig. 7.8.



![Fig. 7.8: Motion capturing](images/Motion_Capturing.png)


	


## Computer Vision Tasks and Approaches

As computer vision is a wide field, there are many groups of tasks that may or may not be relevant in a concrete project. This is a simple grouping of computer vision tasks:

1. *Image acquisition*: Obtaining image data from light-sensitive cameras, ultra-sonic cameras, radars, range sensors, PET/CT/MRI/ultrasound devices, etc. The image data may be 2D or 3D / still images or sequences (video).
1. *Pre-processing*: Preparing the image data for further processing, e.g. by scaling, noise reduction, contrast enhancement, etc. Approaches include filtering and transformation algorithms.
1. *Feature extraction*: Identifying lines, edges, ridges, corners, texture, etc. in images. In use are specific algorithms, e.g., for edge detection. 
1. Segmentation: Identifying image regions of particular interest, e.g. faces in photos. Machine learning approaches are used.
1. *High-level processing*: Application-specific image processing, e.g., classification, image recognition, scene analysis, etc. Machine learning approaches as well as other AI approaches for decision making are used.
1. *Image Generation*: generating images (still or moving) from an internal representation (usually 3D). Specific rendering algorithms are used.



## Services and Product Maps

Fig. 7.9 shows the services map for the area of computer vision.

{width=75%}
![Fig. 7.9: Computer vision services map](images/Computer_Vision_SM.png)

Like for machine learning, there are products available on the library level, the framework level, and the web service level.

- *CV / ML algorithms and libraries*: Algorithms for image pre-processing and feature extraction as well as machine learning libraries.
- *CV / ML APIs*: APIs for interfacing with various ML libraries
- *CV / ML development environments / frameworks*: [IDEs](Integrated_Development_Environments) and frameworks  for experimenting with different computer vision  approaches and configuring solutions.
- *CV web services*: Web services for image search, named entity recognition, etc.
- *CV pre-trained models*: Pre-trained ML models for transfer learning CV tasks. 


Fig. 7.10 shows the product map for computer vision. 

{width=85%}
![Fig. 7.10: Computer vision product map](images/Computer_Vision_PM.png)

[TensorFlow](https://www.tensorflow.org) and [OpenCV](http://opencv.org) are examples for CV / ML libraries. [RapidMiner](https://rapidminer.com) is an example IDE for machine learning. 
[Keras](https://keras.io) is a Python Deep Learning library, interfacing to TensorFlow, CNTK, or Theano.
Examples for CV web services are: [Autokeyword](http://autokeyword.me) and [clarifai](http://www.clarifai.com) for entity recognition, [tineye](https://www.tineye.com) and [Google image search](https://www.google.de) for image retrieval.

More products and details can be found in the appendix.



### Example: OCR with Deep Learning using TensorFlow


[TensorFlow](https://www.tensorflow.org/) is an open source Python library for machine learning. It was developed by members of Google's Machine Intelligence research organization. 

The simple OCR (Object Character Recognition) example is taken from a TensorFlow from [Towards Data Science](https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d) (Yalçın, 2018). 
The task is to recognize digits from images where each image contains exactly one digit.
See Fig. 7.11.

![Fig. 7.11: Sample images with digits (Tensorflow, 2016)](images/MNIST.png)

#### MNIST

The images are taken from the [MNIST](http://yann.lecun.com/exdb/mnist) database of handwritten digits, available for learning computer vision techniques. 
Each image is 28 pixels by 28 pixels. Those pixels can be interpreted as an 28x28 array of numbers. See Fig. 7.12.

![Fig. 7.12: Interpreting an image as array of pixels (Tensorflow, 2016)](images/PixelArray.png)

The array may be flattened as a vector of 28x28 = 784 numbers which will be used as input for machine learning. 55.000 training images are available, all categorized with the digit (0...9) they represent. 
See Fig. 7.13.
The categories are *one-hot encoded*. This means that there are 10 columns in the training data set, one for each digit. If, e.g., the image depicts the digit 5 then in the column for digit 5 there will be a 1 and in all other columns there will be a 0. 

{width=50%}
![Fig. 7.13: Representation of MNIST training data (Tensorflow, 2016)](images/MNIST_Array.png)


#### Deep Learning

Deep learning has become the de-facto standard for image processing and is used and explained in this tutorial. 
See Fig. 7.14 for a typical deep learning network topology for image processing.

{width=75%}
![Fig. 7.14: Deep learning for image processing, adapted from (Ananthram, 2018)](images/Deep_Learning_CV.png)

The input layer of the deep neural network are neurons representing each pixel value of the images to be classified: 784 values in the MNIST example. The output of the network are neurons representing each class, i.e., one neuron for each digit 0...9.
Between input layer and output layer are several *convolutional layers, pooling layers* and *fully connected layers*. Each output of one layer is input to the next layer.  

Convolutional layers are used to extract features from the images, e.g., edges. The idea is to use a small pixel filter (3x3 matrix in Fig. 7.15) which is successively compared with the pixels of the image (5x5 matrix in Fig. 7.15). The comparison is performed simply by computing the dot product: the higher the dot product, the better the match. 
The comparison is performed step by step, eg.,  with a 1 pixel shift at each step. The result is a smaller matrix than the original pixel matrix, a 3x3 matrix in Fig. 7.15. The resulting matrix preserve the relationship between different parts of an image while decreasing the complexity. 

{width=75%}
![Fig. 7.15: Convolutional layer (Yalçın, 2018)](images/Convolutional_Layer.png)

It is common to insert a pooling layer after each convolutional layer. A pooling layer further decreases complexity by considering parts of a matrix (2x2 matrix in Fig. 7.16) and computing a simple aggregation function like maximum of numbers in this part. The resulting matrix is smaller, e.g., 2x2 in Fig. 7.16.



{width=40%}
![Fig. 7.16: Pooling layer(Yalçın, 2018)](images/Pooling_Layer.png)


After successions of convolutional layers and pooling layers for reducing complexity while focusing on relevant features, a set of fully connected layers are used for classification. As the name suggests, in fully-connected layers, each neuron of one layer is connected with all neurons of the next layer. See Fig. 7.17. 


{width=50%}
![Fig. 7.17: Fully connected layers (Yalçın, 2018)](images/Fully_Connected_Layer.png)


#### Keras and TensorFlow

I will now explain parts of the Keras and TensorFlow code from (Yalçın, 2018).

Keras and Tensorflow allow importing and downloading the MNIST dataset directly from their API.

	import tensorflow as tf
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()



It is important to access the shape of the dataset to channel it to the convolutional layers. This is done using the `shape` attribute of `numpy` arrays.

	x_train.shape


The result is  `(55000, 28, 28)`. `55000` represents the number of images in the train dataset and `(28, 28)` represents the size of the image: 28 x 28 pixels.



To be able to use the dataset in Keras, the data must be normalized  as it is always required in neural networks. This can be achieved by dividing the RGB codes by 255. 
Furthermore, the data format required by the API must be met; here, the three-dimensional arrays must be converted to four-dimensional arrays. 

	# Reshaping the array to 4-dims so that it can work with the Keras API
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)
	# Making sure that the values are float so that we can get decimal points after division
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	# Normalizing the RGB codes by dividing it to the max RGB value.
	x_train /= 255
	x_test /= 255




The following code implements the architecture of the simple deep neural network with the following layers:

1. A convolutional layer
2. A max pooling layer
3. A flatten layer to avoid overfitting by disregarding some of the neurons
4. 2 dense layers for classification
5. A flatten layer to convert 2D arrays to a 1D array.


	# Creating a Sequential Model and adding the layers
	model = Sequential()
	model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
	model.add(Dense(128, activation=tf.nn.relu))
	model.add(Dropout(0.2))
	model.add(Dense(10,activation=tf.nn.softmax))


The following code specifies an optimizer and loss function that uses a metric for training. 

	model.compile(optimizer='adam', 
	              loss='sparse_categorical_crossentropy', 
	              metrics=['accuracy'])


Now the model can be trained. The value `epochs` specifies how often all training data shall be used. 

	model.fit(x=x_train,y=y_train, epochs=10)



Finally, you may evaluate the trained model with x_test and y_test using one line of code:

	model.evaluate(x_test, y_test)


The result is an accuracy of 98.5% on the test set. This is quite a good result for such a simple model and only 10 epochs of training. If, however, an error of 0.1% is not tolerable, then the model can be optimized, e.g., by experimenting with  more epochs, different optimizers or loss functions, more layers, different hyperparameters etc. 


The trained model can now be used to predict the digit being depicted in an unknown image. 

	image_index = 4444
	plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
	pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
	print(pred.argmax())

In this example, the digit 9 is returned which is, indeed the correct classification of the image with index 4444. See  Fig. 7.18.

{width=20%}
![Fig. 7.18: Image example (Yalçın, 2018)](images/MNIST_9.png)


Using a ML library like TensorFlow requires a considerably deeper understanding of the algorithms at hand than working with a ML IDE / framework like RapidMiner. However, it allows optimizing an application most specifically.


### Example: Transfer Learning





### Example: Named Entity Recognition with the Autokeyword.me Web Service

Web services for image tagging like [Autokeyword.me](http://autokeyword.me) or [autotag.me](http://autotag.me/) allow classifying and tagging images of any kind. See Fig. 7.13.  

![Fig. 7.13: Tagging example with Autokeyword.me](images/Autokeyword.png)

%% http://autokeyword.me/demo/tags.php?key=common&ftc=&qurl=http%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2F8%2F82%2FJohn_Everett_Millais_-_Isabella.jpg 

In this example, the painting ["Isabella" by John Everett Millais (1829-1896)](https://upload.wikimedia.org/wikipedia/commons/8/82/John_Everett_Millais_-_Isabella.jpg)  is analyzed. Tagging results include concepts like woman and man as well as attributes like attractive and smiling. They are all provided with a degree of probability, indicated by a bar at the circles. For example, the category man is more probable than boy and, indeed, there are men but no single boy depicted in the painting. So, the category "boy" is a false positive. There are also false negatives, e.g., the dogs on the painting were not recognized.

Using an image tagging web service is easy and simply requires invoking an API. However, if the results are not suitable for an application use case, there is no way of optimizing the solution like when using a library or a framework.




## Quick Check


1. Name applications of computer vision
1. What are the main tasks and approaches in computer vision?
1. Which libraries / frameworks / web services can be used for computer vision?
2. What are the advantages / disadvantages of using a web service compared to a library or framework?



