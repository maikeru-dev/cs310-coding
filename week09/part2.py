import time

import numpy as np
from matplotlib import pyplot as plt, pyplot
from pandas import read_csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import Input
import keras.optimizers.schedules as schedules
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam
from keras.losses import Huber
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.activations import swish
from keras.regularizers import l2
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

physical_devices = tf.config.experimental.list_physical_devices("GPU")
for gpu in physical_devices:
    print("got a gpu!")
    tf.config.experimental.set_memory_growth(gpu, True)


ins_file = "unsplit_900x5_Shuf_4prior_0_diff_alignDCT_sent_ins.csv"
labs_file = "unsplit_900x5_Shuf_4prior_0_diff_alignDCT_sent_labs.csv"

scaler_X = PowerTransformer()
scaler_Y = RobustScaler()
print("Reloaded!")
model: Sequential = None

def loadModel(filename):
    global model
    # Load the model from the file
    model = tf.keras.models.load_model(filename)
    print("Model loaded!")

def saveModel(filename):
    global model
    # Save the model to a file
    model.save(filename)
    print("Model saved!")



def read_data(ins_file, labs_file):
    global X_train, X_test, y_train, y_test
    # Read inputs and labels
    # split into input and output columns
    X = read_csv(ins_file, header=None)
    y = read_csv(labs_file, header=None)
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    print(X_train.shape, X_test.shape, y_train.shape,
          y_test.shape)  # pyright: ignore


def modelling(X_train, y_train, X_test, y_test):
    # All model work should be submitted in this function
    # THis includes creation, training, evaluation etc.
    global scaler_X, scaler_Y, model

    tX_train = scaler_X.fit_transform(X_train)
    tX_test = scaler_X.transform(X_test)

    # Scale the target values
    ty_train = scaler_Y.fit_transform(y_train)
    ty_test = scaler_Y.transform(y_test)

    # print("Mean of scaled y_train:", scaler_Y.mean_)  # Should be ~0
    # print("Std of scaled y_train:", scaler_Y.scale_)  # Should be ~1
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)  # Should be (samples, 23)

    n_features = tX_train.shape[1]
    tf.config.threading.set_inter_op_parallelism_threads(16)
    # Create model
    model = Sequential()  # pyright: ignore
    model.add(Input(shape=(n_features,)))
    model.add(Dense(1528, kernel_regularizer=l2(0.01), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Dense(984, kernel_regularizer=l2(0.01), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Dense(578, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Dense(520, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation(swish))
    model.add(Dense(315, use_bias=False))
    model.add(Activation(swish))
    model.add(Dropout(0.3))
    model.add(Dense(23, activation="linear"))

    lr_schedule = schedules.learning_rate_schedule.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9
    )
    model.compile(
        # optimizer=Adam(learning_rate=0.001),
        optimizer=SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True),
        # optimizer=AdamW(learning_rate=0.001),
        loss=Huber(delta=1.0),
        metrics=["mse", "mae"],
    )
    # model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    early_stop = EarlyStopping(
        monitor="loss", patience=5, restore_best_weights=True)
    model.summary()
    model.fit(
        tX_train,
        ty_train,
        epochs=32,
        validation_split=0.2,
        batch_size=64,
        verbose="1",
        callbacks=[early_stop],
    )

    loss = model.evaluate(tX_test, ty_test)

    # Return model at end
    return model


def checkResiduals(X_test, y_test, model):
    global scaler_X, scaler_Y
    y_test = scaler_Y.transform(y_test)
    X_test = scaler_X.transform(X_test)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=50)
    plt.axvline(0, color="red")  # Zero residual line
    plt.show()


def visScatterPlot(X_test, y_test, model):
    global scaler_X, scaler_Y
    y_test = scaler_Y.transform(y_test)
    X_test = scaler_X.transform(X_test)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
    )  # Ideal line
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()


def visualisation(X_test, y_test, model):
    global scaler_Y, scaler_X
    # Scale the test features
    tX_test = scaler_X.transform(X_test)

    # Evaluate (if y was scaled during training, scale it here too)
    ty_test = scaler_Y.transform(y_test)
    mse = model.evaluate(tX_test, ty_test, verbose=1)
    # print("MSE: %.3f, RMSE: %.3f" % (mse, np.sqrt(mse)))

    labelNo = [1500, 5789, 11370, 24, 501, 25999]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, label in enumerate(labelNo):
        row = X_test.iloc[label, :].values.reshape(1, -1)
        trow = scaler_X.transform(row)

        # Predict output (scaled)
        tyhat = model.predict(trow)

        # Inverse transform the prediction
        yhat = scaler_Y.inverse_transform(tyhat)

        # Get actual label (no transformation needed as we want original scale)
        lab = y_test.iloc[label].values

        axes[idx].plot(lab, label="Actual")
        axes[idx].plot(yhat[0], label="Predicted")
        axes[idx].set_title(f"Sample {label}")
        axes[idx].legend()

    plt.tight_layout()
    plt.show()


def extras_view_input_data(X_test, labelNo):
    # Extras, create a figure to look at the input data

    # Extrac data and convert to list
    row = X_test.iloc[labelNo, :]
    # Extract the values only as an array and convert
    row = row.values
    row = row.tolist()

    fig_inps = plt.figure()
    # Add a subplot covering screen
    ax = fig_inps.add_subplot(111)
    # Plot data on subplot
    ax.plot(row)
    # Show on screen6
    plt.show()


def extras_view_label(y_test, labelNo):
    # Display prediction

    lab = y_test.iloc[labelNo, :]
    fig_inps = plt.figure()
    # Add a subplot covering screen
    ax = fig_inps.add_subplot(111)
    # Plot data on subplot
    ax.plot(lab)
    # Show on screen6
    plt.show()


def model():
    global model
    model = modelling(X_train, y_train, X_test, y_test)
