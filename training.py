import tensorflow as tf
from tensorflow import keras

import keras.backend as K
from keras.optimizer_v1 import Adam

from cnnmodel import GenreModel

from datagen import train_generator, vali_generator


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val  = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


model = GenreModel(input_shape=(288,432,4), classes=10)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer = opt, loss='categorical_crossentropy',  metrics=['accuracy', get_f1])

model.fit_generator(train_generator, epochs=70, validation_data=vali_generator)
