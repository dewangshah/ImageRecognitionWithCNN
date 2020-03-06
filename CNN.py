#PART 1 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32,3,3, input_shape= (64,64,3), activation = 'relu'))
#32 - No. of Feature Detectors, 3 - No. of rows of Feature Detector, 3 - No. of rows of Feature Detector
#input_shape =(64,64 - Dimension of Image, 3 - No. of channels of image)

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a second convolutional layer to improve performance
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
#categorical_crossentropy if more than 2 classifications

#PART 2 - Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator

#Image Augmentation - We apply several transformations to the training_set images
train_datagen = ImageDataGenerator(
        rescale=1./255, #Compulsory - This is the feature scaling part of data preprocessing
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) #Feature Scaling of the test_set images

training_set = train_datagen.flow_from_directory(
        'C://Users//Dewang//Desktop//Practice Apps//Deep Learning A-Z//CNN//dataset//training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary') #We have 2 output classes

test_set = test_datagen.flow_from_directory(
        'C://Users//Dewang//Desktop//Practice Apps//Deep Learning A-Z//CNN//dataset//test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier .fit_generator(
        training_set,
        steps_per_epoch=8000, #training_set size
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) #test_set size

"""
Epoch 1/25
8000/8000 [==============================] - 1694s 212ms/step - loss: 0.3595 - accuracy: 0.8324 - val_loss: 0.7412 - val_accuracy: 0.8115- ETA: 8:27 - loss: 0.4282 - accuracy: 0.7942 - ETA: 5:54 - loss: 0.4060 - accuracy: 0.8069
Epoch 2/25
8000/8000 [==============================] - 1765s 221ms/step - loss: 0.1014 - accuracy: 0.9618 - val_loss: 0.6346 - val_accuracy: 0.8068: 8:45 - loss: 0.1180 - accuracy: 0.9551
Epoch 3/25
8000/8000 [==============================] - 1671s 209ms/step - loss: 0.0483 - accuracy: 0.9829 - val_loss: 0.6652 - val_accuracy: 0.8143
Epoch 4/25
8000/8000 [==============================] - 1471s 184ms/step - loss: 0.0352 - accuracy: 0.9880 - val_loss: 0.6988 - val_accuracy: 0.8054: 7:15 - loss: 0.0367 - accuracy: 0.9874
Epoch 5/25
8000/8000 [==============================] - 1481s 185ms/step - loss: 0.0286 - accuracy: 0.9903 - val_loss: 0.5519 - val_accuracy: 0.8044
Epoch 6/25
8000/8000 [==============================] - 1480s 185ms/step - loss: 0.0233 - accuracy: 0.9921 - val_loss: 1.5811 - val_accuracy: 0.8172
Epoch 7/25
8000/8000 [==============================] - 1483s 185ms/step - loss: 0.0202 - accuracy: 0.9934 - val_loss: 0.3769 - val_accuracy: 0.8185
Epoch 8/25
8000/8000 [==============================] - 1484s 185ms/step - loss: 0.0190 - accuracy: 0.9937 - val_loss: 1.1984 - val_accuracy: 0.8028
Epoch 9/25
8000/8000 [==============================] - 1488s 186ms/step - loss: 0.0162 - accuracy: 0.9946 - val_loss: 1.5805 - val_accuracy: 0.8124
Epoch 10/25
8000/8000 [==============================] - 1490s 186ms/step - loss: 0.0154 - accuracy: 0.9951 - val_loss: 2.0508 - val_accuracy: 0.7997
Epoch 11/25
8000/8000 [==============================] - 1482s 185ms/step - loss: 0.0132 - accuracy: 0.9957 - val_loss: 1.8142 - val_accuracy: 0.8106
Epoch 12/25
8000/8000 [==============================] - 1483s 185ms/step - loss: 0.0134 - accuracy: 0.9958 - val_loss: 1.5356 - val_accuracy: 0.8224
Epoch 13/25
8000/8000 [==============================] - 1484s 186ms/step - loss: 0.0121 - accuracy: 0.9959 - val_loss: 2.3848 - val_accuracy: 0.8041
Epoch 14/25
8000/8000 [==============================] - 1481s 185ms/step - loss: 0.0103 - accuracy: 0.9967 - val_loss: 2.6166 - val_accuracy: 0.8080
Epoch 15/25
8000/8000 [==============================] - 1476s 185ms/step - loss: 0.0106 - accuracy: 0.9966 - val_loss: 3.7157 - val_accuracy: 0.8155
Epoch 16/25
8000/8000 [==============================] - 1474s 184ms/step - loss: 0.0105 - accuracy: 0.9967 - val_loss: 3.0560 - val_accuracy: 0.8054
Epoch 17/25
8000/8000 [==============================] - 1478s 185ms/step - loss: 0.0092 - accuracy: 0.9971 - val_loss: 2.2341 - val_accuracy: 0.8104
Epoch 18/25
8000/8000 [==============================] - 1476s 185ms/step - loss: 0.0092 - accuracy: 0.9972 - val_loss: 2.4702 - val_accuracy: 0.8053
Epoch 19/25
8000/8000 [==============================] - 1450s 181ms/step - loss: 0.0091 - accuracy: 0.9972 - val_loss: 2.8316 - val_accuracy: 0.8080
Epoch 20/25
8000/8000 [==============================] - 1400s 175ms/step - loss: 0.0085 - accuracy: 0.9974 - val_loss: 3.5662 - val_accuracy: 0.8081
Epoch 21/25
8000/8000 [==============================] - 1421s 178ms/step - loss: 0.0083 - accuracy: 0.9976 - val_loss: 1.1266 - val_accuracy: 0.8315
Epoch 22/25
8000/8000 [==============================] - 1473s 184ms/step - loss: 0.0081 - accuracy: 0.9976 - val_loss: 1.4197 - val_accuracy: 0.8229
Epoch 23/25
8000/8000 [==============================] - 1516s 189ms/step - loss: 0.0074 - accuracy: 0.9978 - val_loss: 2.8717 - val_accuracy: 0.8090ETA: 16:31 - loss: 0.0066 - accuracy: 0.9980
Epoch 24/25
8000/8000 [==============================] - 1482s 185ms/step - loss: 0.0072 - accuracy: 0.9979 - val_loss: 1.4021 - val_accuracy: 0.7965
Epoch 25/25
8000/8000 [==============================] - 1473s 184ms/step - loss: 0.0072 - accuracy: 0.9979 - val_loss: 2.1564 - val_accuracy: 0.8204

The high accuracy of the training set and the comparatively low accuracy of the test set indicates overfitting of data
""" 

#Part 3 - Making New Predictions

import numpy as np
from keras.preprocessing import image
test_image=image.load_img('C://Users//Dewang//Desktop//Practice Apps//Deep Learning A-Z//CNN//dataset//single_prediction//pug.jpg',
                          target_size=(64,64))

test_image = image.img_to_array(test_image) #Converting 64x64 Image to a 3 dimensional 64x64 array
test_image = np.expand_dims(test_image, axis=0) #Classifier takes 4 dimensional input, which is a batch of 32 3 channel images
#Therefore we pass image by creating a batch and setting the image at the first axis 
result = classifier.predict(test_image)

training_set.class_indices # Gives Output class classification indices

if result[0][0]==1:
    prediction = 'dog'
else:
    predicton ='cat'