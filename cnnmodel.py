import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
                          Flatten, Conv2D, AveragePooling2D,  MaxPooling2D, GlobalMaxPooling2D,
                          Dropout)
from keras.models import Model, load_model
from tensorflow.python.ops.init_ops_v2 import glorot_uniform


def GenreModel(input_shape = (288,432,4),classes=10):

    X_input = Input(input_shape)

    X = Conv2D(8, kernel_size=(3,3), strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(16, kernel_size=(3,3), strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(32, kernel_size=(3,3), strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(64, kernel_size=(3,3), strides=(1,1))(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(128, kernel_size=(3,3), strides=(1,1))(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Flatten()(X)

    X = Dropout(rate=0.3)

    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=10))(X)

    model = Model(inputs=X_input, outputs=X, name='GenreModel')

    return  model