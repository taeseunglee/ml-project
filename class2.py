'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


from keras.models import Model
from keras import optimizers

from keras import backend as K
# dimensions of our images.
img_width, img_height = 64, 64

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '../data/train/'
validation_data_dir = '../data/validation/'
nb_train_samples = 480 # 768
nb_validation_samples = 16 # 64
epochs = 1 # 50
batch_size = 16 # 64

print(img_width, img_height)

def build_model():
	top_model_weights_path = './model/bottleneck_fc_model.h5'
	# build the VGG16 network
	model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
	print('Model loaded.')

	# build a classifier model to put on top of the convolutional model
	print(model.output_shape[1:])
	top_model = Sequential()
	top_model.add(Flatten(input_shape=model.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))


	# note that it is necessary to start with a fully-trained
	# classifier, including the top classifier,
	# in order to successfully do fine-tuning
    # top_model.load_weights(top_model_weights_path)
	model = Model(inputs=model.input, outputs= top_model(model.output))
	print(model.input_shape, model.output_shape)
    # add the model on top of the convolutional base
    # model.add(top_model)

	# set the first 25 layers (up to the last conv block)
	# to non-trainable (weights will not be updated)
	for layer in model.layers[:25]:
		layer.trainable = False

	# compile the model with a SGD/momentum optimizer
	# and a very slow learning rate.
	model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
				  metrics=['accuracy'])

	return model


def save_bottlebeck_features(model = None):
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    
    bottleneck_features_train = model.predict_generator(
            train_generator,
            nb_train_samples // batch_size,
            verbose=0
            )
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
           validation_generator,
           nb_validation_samples // batch_size,
           verbose=0
           )
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

    print(train_generator, validation_generator)

    # fine-tune the model
    model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples)
    print('save bottleneck')
    
    # fine-tune the model
    model.fit_generator(train_generator,
            samples_per_epoch=nb_train_samples,
            epochs=epochs,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)
    print('model fit')

def train_top_model(model):
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    print(train_data.shape, train_labels.shape, train_data.shape[1:])
    print(validation_data.shape, validation_labels.shape)


    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

     # load an image from file
    image = load_img('./data/train/pants/cargo/pants_cargo00000.jpg', target_size=(img_width, img_height))

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))

   

    return train_data.shape[1:]

def test(input_shape):
    base_model = applications.VGG16(include_top=False, weights='imagenet')

    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_shape))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    top_model.load_weights(top_model_weights_path)

    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

    # load an image from file
    image = load_img('../data/train/pants/cargo/pants_cargo00000.jpg', target_size=(img_width, img_height))

    # convert the image pixels to a numpy array
    image = img_to_array(image)
    print(image)

    # reshape data for the model
    # image = image.reshape((1, img_width, 2, 512))

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))

def hello(model):
     # load an image from file
    image = load_img('../data/train/pants/pants00000.jpg', target_size=(img_width, img_height))

    # convert the image pixels to a numpy array
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # prepare the image for the VGG model
#    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)
    print(yhat)
    print(type(yhat))

    # convert the probabilities to class labels
    label = decode_predictions(yhat, top=1)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))


model = build_model()
hello(model)
#save_bottlebeck_features(model)
# input_shape = train_top_model(model)
