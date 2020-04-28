[//]: # (Image References)

[image1]: figures/fig_dataset_training.png  "Distribution of images per breed"
[image2]: figures/fig_dataset_sample_subset_categories.png "Sample of dataset images"
[image3]: figures/fig_dataset_sample_subset_category_variation.png "Sample of dataset images"
[image4a]: figures/dog_prev_0_4019.jpeg "Sample of augmented images"
[image4b]: figures/dog_prev_0_7317.jpeg "Sample of augmented images"
[image4c]: figures/dog_prev_0_8328.jpeg "Sample of augmented images"
[image5]: figures/fig_fit_history.png "Model training history"
[image6]: figures/fig_confusion_matrix.png "Confusion matrix"

<a id='top'></a>
# Description

In this project, [transfer learning](http://cs231n.github.io/transfer-learning/) using a [Convolutional Neural Network](https://en.wikipedia.org/wiki/CNNConvolutional_neural_network) (CNN) trained on the [ImageNet](http://www.image-net.org/) datset is applied to build a classification model for the breed of a dog in a image. The classification task is implemented in a python packaged and a flask-based web app is provided to allow users to interact with the dog breed classification model.

The definition of the project is presented in the section [Project Defintion](#section1). The [data exploration](#section2) of the image data sets, algorithm [methodology](#section3) and [results](#section4) of classification are also presented. Some discussinons of the CNN model is presented in [Conclusions](#section5). [Instructions](#section6) are provided on how to deploy the application.


* [Disclaimer](#section0)
* [Project Definition](#section1)
* [Data Exploration](#section2)
* [Methodology](#section3)
* [Results](#section4)
* [Conclusions](#section5)
* [Instructions](#section6)


---
<a id='section0'></a>
## Disclaimer
This project is based on the very popular deep learning algorithm for [dog breed classification](https://github.com/udacity/dog-project) from  [Udacity](https://www.udacity.com/). The implentation of the classification algorithm in this repository was motivated by the instructions in the [python notebook](https://github.com/udacity/dog-project/blob/master/dog_app.ipynb) by [Chris Gearhart](https://github.com/cgearhart) and  [Luis Serrano](https://github.com/luisguiserrano).


---
<a id='section1'></a>
## Project Definition

In this project, the problem consists in identifying the breed of a dog from an image. This is a very challenging problem given the large similarity between different dog breeds. This is a hard task even for non-experts due to the large number of breeds and the similarities among some breeds. Besides, for the same breed, variabilities like different colors and shade of the dog coat.

The problem is solved by using imaging processing techniques and deep learning algorithms. The task consists in detecting a dog in a image and then classify the dog breed. If a dog is not detected in the image, the algorithm detects if a human face is present, and, if so, the most resembling dog breed to the human face is computed.

[Back to top](#top)

---
<a id='section2'></a>
## Data Exploration

The image data set consists of 8351 images of dogs and 13233 images of humans available from the [dog project](https://github.com/udacity/dog-project). The dog images are splitted into, training, validation and test sets with internal directory tree for each one of the 133 dog categories. The split is as follows:
```
There are 133 total dog categories.
There are 8351 total dog images.

There are 6680 training dog images.
There are 835 validation dog images.
There are 836 test dog images.
```

The distribution of images per dog breed is highly umbalanced with an average of 50 images per breed in the training set:
```
Images per breed in training set:
 Min: 26
 Mean: 50.2
 Max: 77
 ```
![Fig1][image1]

A sample of the 4 images for a subset of the dog breeds is shown below:

![Fig2][image2]

The images below show a sample of 4 images for the dog breed *Alaskan malamute*.

![Fig3][image3]


[Back to top](#top)

---
<a id='section3'></a>
## Methodology

### Data Preprocessing

The dog breed images dataset were preprocessed to resize the images to 224x224 pixels. In addition, the pre-processing function `keras.applications.resnet50.preprocess_input` was applied. It reorders the channels from RGB to BGR and normalizes the pixel values.

Furthermore, image augmentation was applied with `keras.preprocessing.image.ImageDataGenerator` as the number of images per dog breed is rather low. Last, feature extratation from the images using ResNet50 was performed and the bottleneck data stored for faster train of the model. For the image augmentation, rotation, translations, shear and zoom transformations were applied. Below a sample for augmented images is presented.

![Fig4a][image4a] | ![Fig4b][image4b] | ![Fig4c][image4c]


### Implementation

 The dog breed application uses three trained models for the tasks of human face detection, dog detection and dog breed classification as detailed below.

**Human face detector:**
human faces in images are detected using the OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html). OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). In particular, the model `haarcascade_frontalface_alt.xml` was used as it was observed to best perform on the human images dataset (see [results](#section4)).

**Dog detector:**
dogs in images are detected usin a pre-trained ResNet-50 model available in Keras. The model has been trained on ImageNet for image classification of 1000 categories of objects. In particular, the list of categories contains dogs. The subset of categories in the range 151-268 in ImageNet corresponds to dogs and were used to implement a binary classifier defining a simple and very effecient dog detector model (see [results](#section4)).

**Dog breed classifier:**
Transfer learning was applied to create a CNN based on the pre-trained ResNet-50 CNN on the ImageNet dataset. The ResNet-50 model performs as a feature extractor, where the last convolutional output of ResNet-50 is connected to a Keras sequential model pooling layer, dropout layer, and a fully connected layer with 133 nodes to classify the dog breeds. The output layer is activated with a softmax. A dropout layer was necessary to reduce overfittig issues (see [results](#section4)). The model architure is summarized below:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_5 ( (None, 2048)              0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 200)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 133)               272517    
=================================================================
Total params: 272,517
Trainable params: 272,517
Non-trainable params: 0
```

[Back to top](#top)

---
<a id='section4'></a>
## Results

### Human face detector efficiency

The accuracy for detection of human faces by the Haar feature-based cascade classifiers were tested in a sample of 100 images of humans and 100 images of dogs. The results are presented below.

- `haarcascade_frontalface_default.xml`
    ```
    Ratio of face detection for humans: 0.99
    Ratio of face detection for dogs: 0.54
    ```
- `haarcascade_frontalface_alt.xml`
    ```
    Ratio of face detection for humans: 0.99
    Ratio of face detection for dogs: 0.12
    ```
- `haarcascade_frontalface_alt2.xml`
    ```
    Ratio of face detection for humans: 0.99
    Ratio of face detection for dogs: 0.19
    ```
- `haarcascade_frontalface_alt_tree.xml`
    ```
    Ratio of face detection for humans: 0.57
    Ratio of face detection for dogs: 0.01
    ```
Ideally, the classifier would have 100% efficiency in detecting a face in the human images and would detect no face for the dog images. For this project, the classifier `haarcascade_frontalface_alt.xml` was choose as it provides the best results of high ratio of face detection in human images and lower ratio of face detecions in dog images.

### Dog detector efficiency

The efficiency of the dog detector for images of humans and dogs is given below:
```
Ratio of dog detection for humans: 0.0
Ratio of dog detection for dogs: 0.99
```
The model shows a high efficiency for detecting dogs in the dog images.

### Model Evaluation and Validation

The model complexity graph is illustrated below for the training using the original dataset and a dropout ratio of 0.5.

![Fig5][image5]

The evaluation scores of the model on the test set were:
```
Accuracy: 0.8182  
Balanced accuracy score: 0.8027
F1 (micro): 0.8182
F1 (macro): 0.8047
F1 (weighted): 0.8183
```

The image below presents the performance of the model on the individual dog breeds.

![Fig6][image6]


[Back to top](#top)

---
<a id='section5'></a>
## Conclusion

The problem of dog breed classification was performed by applying a chain of Neural Networks to detect dogs or human face and then to classify the dog breed. Pre-trained models were used to implement the algorithms for human face and dog detection in images. For the dog breed classifier, transfer learning was applied to develope a CNN to classify the dog breeds trained on a set of 8351 images of dogs of different breeds. Data augmentation was applied to tackle the problem of small samples per category. However, performance of the model with augmented data was not improved. The full machine learning pipeline was implemented to augment data, train model and deploy. The classifier has an accuracy above 80%. Extending the dataset might help to improve further the model accuracy.


[Back to top](#top)

---
<a id='section6'></a>
## Instructions 

**Note:** this repository is based on the original repository from [Udacity](https://github.com/udacity/dog-project). It has been tested only in a local linux machine without using GPU. Plaeas, refere to the [original repository](https://github.com/udacity/dog-project) for further instructions on how to run the application in a different set up.

1. Clone the repository and navigate to the project's directory.

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) and extract the content to the sub-directory `data/dogImages`:
   ```
   wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
   unzip dogImages.zip
   mv dogImages data/
   ```

3. (Optional) Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) and extract the content to the sub-directory `data/lfw`. This dataset can be used to test the performance of the face detector. However, the dataset is neither required to train the CNN for dog breed classification, neither to deploy the web-app.
   ```
   wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
   unip lfw.zip
   mv lfw data/
   ```

4. Set up a virtual environment and install the python packages requirements.
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

5. Switch Keras backend to TensorFlow
    ```
    KERAS_BACKEND=tensorflow python -c "from keras import backend"
    ```

6. Geneate the bottleneck features for the dog dataset:
   ```
    
   python dogclf/generateBottleneckFeatures.py
   ```

7. Train the CNN classifier:
    ```
    python dogclf/trainClassifier.py
    ```

8.  Deploy the web-app:
    ```
    python webapp.py
    ```
