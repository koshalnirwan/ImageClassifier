# ImageClassifier
 Classify Images of Cats and Dogs

The project is based on Convolutional Neural Networks (CNN). It classifies
the images of Cats and Dogs.

1. 'predictions.py' is a python file that uses saved model to predict the image
2. 'myCNN.model' is a model file that contains the trained model
3.  Model file is developed by training Convolutional Neural Networks on cats and dog images
4.  Model optimization is done using Tensorboard
5.  Keras is the Library used for Deep Learning here
6.  The test Accuracy of model is 80 percent.

## Dataset
Dataset consists of two sets. Training set contains 8000 images each for cats and dogs, whereas test set contains 1500 images each for dogs and cats. So total of 16000 images for training model and 3000 images for testing the model. 

## Data Preprocessing
Preprocessing is done as not all the images are of same shape and size. This is done using Python's OpenCV Library. 

1. Load Original Image 

<img src="Screenshots/cat_org.JPG" width=250) 
<img src="Screenshots/dog_org.JPG" width=350>
   
2. Convert images into gray scale. This is because we are only concerned for the patterns in images and not the color. This is also a memory efficient step.
3. Resize images so that they all are of same size and can be operated on smoothly
4. Convert the image data into a numpy array to feed it to the model



