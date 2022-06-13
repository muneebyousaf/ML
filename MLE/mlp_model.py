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



def mlp_train (X_train, X_test, y_train, y_test,n_features):
	
    
    print(" mlp training has been called")
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






