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
train_data_dir = 'data/train/' # data for training model
validation_data_dir = 'data/validation/' # data for validating model
nb_train_samples = 3200 # the number of train samples 
nb_validation_samples = 340 # the number of validation samples
epochs = 75
batch_size = 20

def build_model():
	# build the VGG16 network
	model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
	print('Model loaded.')

	# build a classifier model to put on top of the convolutional model
	print(model.output_shape[1:])
	top_model = Sequential()
	top_model.add(Flatten(input_shape=model.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(3, activation='softmax')) # 

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
	model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-4),
				  metrics=['accuracy'])

#    model.compile(optimizer='rmsprop',
#                  loss='binary_crossentropy', metrics=['accuracy'])
 
	return model


def save_bottlebeck_features(model = None):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
#    model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    
#    for i in train_generator:
#        for j in i:
#            print(j.shape, type(j), len(j))
#            break

    '''
    bottleneck_features_train = model.predict_generator(
            generator,
            nb_train_samples // batch_size,
            verbose=0
            )

    print('bottleneck train : ', bottleneck_features_train)
    '''
#    np.save(open('bottleneck_features_train.npy', 'w'),
#            bottleneck_features_train)
#    np.save('bottleneck_features_train.npy', bottleneck_features_train)
#    train_data = np.load('bottleneck_features_train.npy')
    
    validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # bottleneck_features_validation = model.predict_generator(
    #        generator, nb_validation_samples // batch_size)
    # np.save('bottleneck_features_validation.npy', bottleneck_features_validation)'''
    # validation_data = np.load('bottleneck_features_validation.npy')

    print(train_generator, validation_generator)

    # fine-tune the model
    model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples,
            epochs=epochs,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)
    print('save bottleneck')


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    print(train_data.shape, train_labels.shape, train_data.shape[1:])
    print(validation_data.shape, validation_labels.shape)

#    model = Sequential()
#    model.add(Flatten(input_shape=train_data.shape[1:]))
##    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1, activation='sigmoid'))

#    model.compile(optimizer='rmsprop',
#                  loss='binary_crossentropy', metrics=['accuracy'])

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
    image = load_img('./data/train/pants/cargo/pants_cargo00000.jpg', target_size=(img_width, img_height))

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

model = build_model()
# model = []
save_bottlebeck_features(model)
input_shape = train_top_model()
test(input_shape)
