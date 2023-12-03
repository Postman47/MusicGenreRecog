import json
import numpy as np
from ann_visualizer.visualize import ann_viz
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from keras_visualizer import visualizer

DATASET_PATH = "data2.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epochs")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data(DATASET_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary

    visualizer(model, format='png', view=True)
    return model

def build_model_2(input_shape):

    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.002), bias_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.002), bias_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.002), bias_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))


    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):

    # add a dimension to input data for sample
    X = X[np.newaxis, ...]

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, predicted label: {}".format(y, predicted_index))

if __name__ == "__main__":
    # load data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model_2(input_shape)

    # ann_viz(model, view=True)
    visualizer(model, view=True)

    #compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer = optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # train network
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=70, batch_size=32)

    # plot accuracy and error over the  epochs
    plot_history(history)

    model.save("model_cnn_512_512_128_lr=0001_332_v3_biasall")
