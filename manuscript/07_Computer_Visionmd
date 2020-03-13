

# Computer Vision

Computer vision is a wide field of AI. It is all about processing images: still images and moving images (videos) / analyzing and generating. Relevant questions are:

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

- Generating images from examination procedures like [PET](Positron Emission Tomography)/[CT](Computed Tomography), [MRI](Magnetic Resonance Imaging), ultrasound images (2D and 3D) for inspection by doctors
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


![Fig. 7.9: Computer vision services map](images/Computer_Vision_SM.png)

Like for machine learning, there are products available on the library level, the framework level, and the web service level.

- *Computer vision / machine learning algorithms and libraries*: Algorithms for image pre-processing and feature extraction as well as machine learning libraries.
- *Computer vision development environments / frameworks*: [IDEs](Integrated Development Environments) and frameworks  for experimenting with different computer vision  approaches and configuring solutions.
- *Computer vision web services*: Web services for image search, named entity recognition, etc.


Fig. 7.10 shows the product map for computer vision. 

![Fig. 7.10: Computer vision product map](images/Computer_Vision_PM.png)

[TensorFlow](https://www.tensorflow.org/) and [OpenCV](http://opencv.org/) are examples for Computer vision / machine learning libraries. RapidMiner is an example IDE for machine learning. Finally these are examples for computer vision web services: [Autokeyword](http://autokeyword.me/) and [clarifai](http://www.clarifai.com/) for entity recognition, [tineye](https://www.tineye.com) and [Google image search](https://www.google.de) for image retrieval.

More products and details can be found in the appendix.



### Example: OCR with the TensorFlow Library


[TensorFlow](https://www.tensorflow.org/) is an open source  library for machine learning. It was developed by members of Google's Machine Intelligence research organization. 

The simple OCR (Object Character Recognition) example is taken from a [TensorFlow tutorial](https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html). 
The task is to recognize digits from images where each image contains exactly one digit.
See Fig. 7.11.

![Fig. 7.11: Sample images with digits](images/MNIST.png)

The images are taken from the [MNIST](http://yann.lecun.com/exdb/mnist/) database of handwritten digits, available for learning computer vision techniques. 
Each image is 28 pixels by 28 pixels. Those pixels can be interpreted as an 28x28 array of numbers; see Fig. 7.12.

![Fig. 7.12: Interpreting an image as array of pixels](images/PixelArray.png)

The array may be flattened as a vector of 28x28 = 784 numbers which will be used as input for machine learning. 55.000 training images are available, all categorized with the digit (0...9) they represent. As machine learning algorithm, a so-called [Softmax regression](http://neuralnetworksanddeeplearning.com/chap3.html#softmax) is used. 

The sample implementation is in [Python](https://www.python.org/). 
The model itself is implemented using the TensorFlow Softmax library implementation as follows.


    y = tf.nn.softmax(tf.matmul(x, W) + b)

In order to train the model, a cost function needs to be defined, here: cross-entropy.

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y_ * tf.log(y), 
        reduction_indices=[1]))

For training the model, backpropagation with a gradient descent optimization algorithm is used.

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


After training, the model needs to be evaluated. Here, the accuracy is computed. 

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


The accuracy of the model with 55.000 sample cases is 92%. Is this good? Well, for some applications it might be sufficient but for many application scenarios, a higher accuracy is required. Working with more complex models allows increasing the accuracy to 99% and higher. 

Using a machine learning library like TensorFlow requires a considerably deeper understanding of the algorithms at hand than working with a machine learning IDE / framework like RapidMiner. However, it allows optimizing an application most specifically.



## Example: Named Entity Recognition with the Autokeyword.me Web Service

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



