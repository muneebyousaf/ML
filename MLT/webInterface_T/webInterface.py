import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
# Load MNIST digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshape data to (n_samples, height, width, n_channel)
X_train = np.expand_dims(X_train, axis=3).astype("float32")
X_test = np.expand_dims(X_test, axis=3).astype("float32")
# One-hot encode the output
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# LeNet5 model
model = Sequential([
Conv2D(6, (5,5), activation="tanh",
input_shape=(28,28,1), padding="same"),
AveragePooling2D((2,2), strides=2),
Conv2D(16, (5,5), activation="tanh"),
AveragePooling2D((2,2), strides=2),
Conv2D(120, (5,5), activation="tanh"),
Flatten(),
Dense(84, activation="tanh"),
Dense(10, activation="softmax")
])
# Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)