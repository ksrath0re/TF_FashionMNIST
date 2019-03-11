from __future__ import absolute_import, print_function, division

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_data = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_data.load_data()

#The shape attribute for numpy arrays returns the dimensions of the array.
#If Y has n rows and m columns, then Y.shape is (n,m). So Y.shape[0] is n.
print (training_images[0].shape)
#print (training_images[0])

#Label 	Class
#0 	    "T-shirt/top",
#1 	    "Trouser",
#2 	    "Pullover",
#3 	    "Dress",
#4 	    "Coat",
#5 	    "Sandal",
#6 	    "Shirt",
#7 	    "Sneaker",
#8 	    "Bag",
#9 	    "Ankle boot",

#Each image is mapped to a single label. Since the class names are not included with the dataset,
#store them here to use later when plotting the images

classes = ["T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"]

#Details of Data Set

print("Shape of Training Images : ", training_images.shape, " and of training labels : ", training_labels.shape)
print("Similarly, Shape of test Images : ", test_images.shape, " and of test labels : ", test_labels.shape)

# inspect the first image in the training set, pixel values fall in the range of 0 to 255

plt.figure()
#plt.imshow(training_images[0])
plt.grid(False)
#plt.show()

#Let's normalize it by dividing by 255

training_images = training_images/255
test_images = test_images/255

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(classes[training_labels[i]])

#plt.show()

#Setup the layers and build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#add extra settings like loss function and metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Now train the model using training examples and labels
model.fit(training_images, training_labels, epochs=3)

#Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test Accuracy ', test_accuracy)
#It turns out, the accuracy on the test dataset is a little less than the accuracy on the training dataset.
#This gap between training accuracy and test accuracy is an example of overfitting

predictions = model.predict(test_images)

print(predictions[0], ' and corresponding class is ', np.argmax(predictions[0]))
print('Actual Label :', test_labels[0])
print('Is our predictions correct : ')
print('Yes') if np.argmax(predictions[0]) == test_labels[0] else print('No')
