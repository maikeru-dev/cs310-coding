import tensorflow.python.distribute.input_lib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

data_path = "../week08/"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

# Dataset preparation
train_input = np.array([np.array(d[1:]) for d in train_data])
# Separating the labels from the image
train_label = np.array([int(d[0]) for d in train_data])

test_input = np.array([np.array(d[1:]) for d in test_data])
# Separating the labels from the image
test_label = np.array([int(d[0]) for d in test_data])


def modelling(train_input, train_label, test_input, test_label):
    # All model work should be submitted in this function
    # This includes creation, training, evaluation etc.

    # determine the number of input features
    n_samples, n_features = train_input.shape

    # Create model
    model: Sequential = Sequential()  # pyright: ignore
    model.add(Dense(25, input_dim=n_features, activation="relu"))
    model.add(Dense(15, input_dim=n_features, activation="relu"))
    model.add(Dense(10, activation="sigmoid"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    model.fit(train_input, train_label, epochs=8, batch_size=32, verbose="1")
    loss, acc = model.evaluate(test_input, test_label)  # pyright: ignore
    print("Accuracy: %.3f", acc)
    # Return model at end
    return model


def visualisation(test_input, test_label, model):
    labelchoice = 0

    predictData = np.array(test_input[labelchoice])[None, ...]
    predictLabel = test_label[labelchoice]

    # make prediction
    yhat = model.predict(predictData)

    print("actual: ", predictLabel)

    # Display prediction
    print("Predicted: %s (class=%d)" % (yhat, np.argmax(yhat)))
    fig = plt.figure()
    plt.plot(yhat[0])
    plt.show()


modelling(train_input, train_label, test_input, test_label)
