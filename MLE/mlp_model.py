import global_data
from tabnanny import verbose
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping
import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask
import json
from app import app


class ProgressCallback(Callback):
	def on_train_begin(self, logs=None):
		global_data.train_status["running"] = True
		global_data.train_status["epoch"] = 0
	def on_train_end(self, logs=None):
		global_data.train_status["running"] = False
	def on_epoch_begin(self, epoch, logs=None):
		global_data.train_status["epoch"] = epoch
		global_data.train_status["batch"] = 0
	def on_epoch_end(self, epoch, logs=None):
		global_data.train_status["last epoch"] = logs
	def on_train_batch_begin(self, batch, logs=None):
		global_data.train_status["batch"] = batch
	def on_train_batch_end(self, batch, logs=None):
		global_data.train_status["batch metric"] = logs

def mlp_train(X_train,y_train,X_test,y_test,n_features):
    
    activation = global_data.model_data["activation"]
    # define model
    activation = global_data.model_data["activation"]
    model = Sequential()
    model.add(Dense(10, activation=activation, kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(8, activation=activation, kernel_initializer='he_normal'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
		optimizer=global_data.model_data["optimizer"],
		metrics=["accuracy"])
    earlystop = EarlyStopping(monitor="val_loss", patience=3,restore_best_weights=True)
    history = model.fit(
				X_train, y_train, validation_data=(X_test, y_test),
				epochs=global_data.model_data["epochs"],
				batch_size=global_data.model_data["batchsize"],
				verbose=0, callbacks=[earlystop,ProgressCallback()])
    return model, history




@app.callback(Output(component_id="epochdisplay", component_property="children"),
	Input(component_id="epochs", component_property="value"))


def update_epochs(value):
	global_data.model_data["epochs"] = value
	return f"Epochs: {value}"

@app.callback(Output("batchdisplay", "children"),
	Input("batchsize", "value"))
def update_batchsize(value):
	global_data.model_data["batchsize"] = value
	return f"Batch size: {value}"


@app.callback(Output("activationdisplay", "children"),
	Input("activation", "value"))

def update_activation(value):
	return f"Activation: {value}"


@app.callback(Output("historyplot", "figure"),
	Input("train", "n_clicks"),
	State("activation", "value"),
	State("optimizer", "value"),
	State("epochs", "value"),
	State("batchsize", "value"),
	prevent_initial_call=True)


def train_action(n_clicks, activation, optimizer, epoch, batchsize):
	global_data.model_data.update({
		"activation": activation,
		"optimizer": optimizer,
		"epcoh": epoch,
		"batchsize": batchsize,
	})

	model, history = train()
	global_data.model_data["model"] = model # keep the trained model
	history = pd.DataFrame(history.history)
	fig = px.line(history, title="Model training metrics")
	fig.update_layout(xaxis_title="epochs",
		yaxis_title="metric value", legend_title="metrics")
	return fig 
@app.callback(Output("progressdisplay", "children"),
	Input("trainprogress", "n_intervals"))

def update_progressdisplay(n):
	return json.dumps(global_data.train_status, indent=4)