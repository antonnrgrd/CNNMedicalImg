import keras

from keras.preprocessing.image import ImageDataGenerator
# Generates multiple data from a single picture. Improves on the generalization.
from keras.models import Sequential

#from keras.layers import MaxPooling2D
# Used to extract features of the image. MaxPooling helps reduce the size of the picture.

from keras.layers import Activation, Dropout, Flatten

from keras import backend as K

import numpy as np

from keras.preprocessing import image

# import tensorflow as tf

import custom_layer
import custom_conv2d
import txt_generator
import custom_pool

# ___________________________________________ BELOW WE DO DATA AUGMENTATION ______________________________________


img_width, img_height = 224, 224  # The width and height we will scale images to.
# img_width, img_height = 150, 150	#We get some memory problems, when we resize the image to be 150*150


train_data_dir = '../data/filtered'  # Points to folder with 2 subfolders in it (healthy/pneumonia).
validation_data_dir = '../data/validation'  # Points used for validation, again 2 subfolders.
txt_validation_data_dir = '../data/encoded'


nb_train_samples = 1000
nb_validation_samples = 100

epochs = 10  # Number of times you give the network the same batch
batch_size = 5  # We get some problems with 224*224 pic, can maybe be changed.

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

'''
ImageDataGenerator generates extra images from the original images, by scaling, rotating etc.
'''
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



#test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='binary')


#validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
#                                                        batch_size=batch_size, class_mode='binary')

validation_generator = txt_generator.CustomDataGenerator(txt_validation_data_dir,
                                                        batch_size = batch_size,
                                                        dim = (img_width, img_height),
                                                        n_channels = 3,
                                                        should_shuffle = True)

# __________________________________________ BELOW IS THE NEURAL NETWORK ITSELF ________________________________________

# A callback that will save the parameters of the model achieving greatest validation success out of all epochs.
# Parameters are saved at the end of each epoch
checkpoint_filepath = 'pneumonia_classifier.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    verbose=1,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
    )


model = Sequential()
print(input_shape)
model.add(custom_conv2d.Conv2D(32, (3, 3),
                 input_shape=input_shape))  # Convolutional network extract features from the images. Search for 32 features, search by 3*3 pixel matrix.
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))  # Reduce size without losing features.
model.add(custom_pool.CustomPooling2D(pool_size=(2,2)))

model.add(custom_conv2d.Conv2D(32, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(custom_pool.CustomPooling2D(pool_size=(2,2)))

model.add(custom_conv2d.Conv2D(64, (3, 3)))  # We can add extra convolutional layers to increase accuracy.
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(custom_pool.CustomPooling2D(pool_size=(2,2)))

model.add(Flatten())  # Flatten image from 2d image to 1d

model.add(custom_layer.Dense(64))
model.add(Activation('relu'))

model.add(custom_layer.Dense(64))	#This extra Dense layer seems to improve by around 5 % 
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(custom_layer.Dense(1))
model.add(Activation('sigmoid'))  # Sigmoid since we have 1 output node, and want a binary probability.

model.summary()

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=0.0005), metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs, callbacks=[model_checkpoint_callback],
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

# We load the best weights from callback checkpoint
model.load_weights(checkpoint_filepath)

#model.save_weights('pneumonia_classifier.h5')

# ____________________________________  BELOW WE JUST TEST 6 RANDOM PICTURES ___________________________________________
# __________ These are supposed to come out 'healthy, healthy, healthy, pneumonia, pneumonia, pneumonia' _______________

img_pred1 = image.load_img('../data/IM-0135-0001.jpeg', target_size=(224, 224))
img_pred1 = image.img_to_array(img_pred1)
img_pred1 = np.expand_dims(img_pred1, axis=0)

img_pred2 = image.load_img('../data/IM-0292-0001.jpeg', target_size=(224, 224))
img_pred2 = image.img_to_array(img_pred2)
img_pred2 = np.expand_dims(img_pred2, axis=0)

img_pred3 = image.load_img('../data/IM-0309-0001.jpeg', target_size=(224, 224))
img_pred3 = image.img_to_array(img_pred3)
img_pred3 = np.expand_dims(img_pred3, axis=0)

img_pred4 = image.load_img('../data/person7_bacteria_29.jpeg', target_size=(224, 224))
img_pred4 = image.img_to_array(img_pred4)
img_pred4 = np.expand_dims(img_pred4, axis=0)

img_pred5 = image.load_img('../data/person23_bacteria_78.jpeg', target_size=(224, 224))
img_pred5 = image.img_to_array(img_pred5)
img_pred5 = np.expand_dims(img_pred5, axis=0)

img_pred6 = image.load_img('../data/person529_bacteria_2229.jpeg', target_size=(224, 224))
img_pred6 = image.img_to_array(img_pred6)
img_pred6 = np.expand_dims(img_pred6, axis=0)

# Below we run the model on an example.
rslt1 = model.predict(img_pred1)
print(rslt1)

rslt2 = model.predict(img_pred2)
print(rslt2)

rslt3 = model.predict(img_pred3)
print(rslt3)

rslt4 = model.predict(img_pred4)
print(rslt4)

rslt5 = model.predict(img_pred5)
print(rslt5)

rslt6 = model.predict(img_pred6)
print(rslt6)

if (rslt1[0][0] == 1):
    print("Pneumonia")
else:
    print("Healthy")

if (rslt2[0][0] == 1):
    print("Pneumonia")
else:
    print("Healthy")

if (rslt3[0][0] == 1):
    print("Pneumonia")
else:
    print("Healthy")

if (rslt4[0][0] == 1):
    print("Pneumonia")
else:
    print("Healthy")

if (rslt5[0][0] == 1):
    print("Pneumonia")
else:
    print("Healthy")

if (rslt6[0][0] == 1):
    print("Pneumonia")
else:
    print("Healthy")
