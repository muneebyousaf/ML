from flask import Flask
from dash import Dash, html, dcc
# default values
model_data = {
"activation": "relu",
"optimizer": "adam",
"epochs": 100,
"batchsize": 32,
}


server = Flask("mlm")
app = Dash(server=server)
app.layout = html.Div(
	id="parent",
	children=[
		html.H1(
			children="LeNet5 training",
			style={"textAlign": "center"}
		),
		html.Div(
			className="flex-container",
			children=[
				html.Div(children=[
					html.Div(id="activationdisplay", children="Activation:"),
					dcc.Dropdown(
						id="activation",
						options=[
							{"label": "Rectified linear unit", "value": "relu"},
							{"label": "Hyperbolic tangent", "value": "tanh"},
							{"label": "Sigmoidal", "value": "sigmoid"},
						],
						value=model_data["activation"]
					)
				]),
				html.Div(children=[
					html.Div(id="optimizerdisplay", children="Optimizer:"),
					dcc.Dropdown(
						id="optimizer",
						options=[
							{"label": "Adam", "value": "adam"},
							{"label": "Adagrad", "value": "adagrad"},
							{"label": "Nadam", "value": "nadam"},
							{"label": "Adadelta", "value": "adadelta"},
							{"label": "Adamax", "value": "adamax"},
							{"label": "RMSprop", "value": "rmsprop"},
							{"label": "SGD", "value": "sgd"},
							{"label": "FTRL", "value": "ftrl"},
						],
						value=model_data["optimizer"]
					),
				]),
				html.Div(children=[
					html.Div(id="epochdisplay", children="Epochs:"),
					dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
					value=model_data["epochs"], id="epochs"),
				]),
				html.Div(children=[
					html.Div(id="batchdisplay", children="Batch size:"),
					dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
						value=model_data["batchsize"], id="batchsize"),
				]),
			]
		),
		html.Button(id="train", n_clicks=0, children="Train"),
	]
)
app.run()
'''
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

for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

print(model.summary())
# Train the model


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

'''
