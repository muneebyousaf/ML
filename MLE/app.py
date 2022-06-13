
from tabnanny import verbose
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping
import plotly.express as px
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from flask import Flask
import json
import global_data
import os 
from urllib.parse import quote as urlquote
import base64
import dash_table_experiments as dt
import io
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import mlp_model
from mlp_model import *
import dash_table as dt


fig = px.bar(
    x=["a","b","c"], y=[1,3,2], # replace with your own data source
    title="sample figure", height=625)

server = Flask("mlm")
app = Dash(server=server)



app.layout = html.Div(
	id="parent",
	children=[
        
		html.H1(
			children="Automatic Vulnerabiliy Assesment" ,
			style={"textAlign": "center"}
		),
       
       
        html.Div(
            
            className="new-container",
            children=[
            
			html.Div(id="modelDisplay", children=" Select Model:"),
				dcc.Dropdown(
					id="model",
					options=[
					    {"label": "MLP", "value": "mlp"},
						{"label": "CNN", "value": "cnn"},
						
						],
						value=global_data.model_data["model_type"]
					),
				]),
       
        html.H3(
			children="Model training attributes:" ,
			
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
						value=global_data.model_data["activation"]
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
						value=global_data.model_data["optimizer"]
					),
				]),
				html.Div(children=[
					html.Div(id="epochdisplay", children="Epochs:"),
					dcc.Slider(1, 200, 1, marks={1: "1", 100: "100", 200: "200"},
						value=global_data.model_data["epochs"], id="epochs"),
				]),
				html.Div(children=[
					html.Div(id="batchdisplay", children="Batch size:"),
					dcc.Slider(1, 128, 1, marks={1: "1", 128: "128"},
						value=global_data.model_data["batchsize"], id="batchsize"),
				]),
			]
		),
		html.Button(id="train", n_clicks=0, children="Train"),
		html.Pre(id="progressdisplay"),
		dcc.Interval(id="trainprogress", n_intervals=0, interval=1000),
		dcc.Graph(id="historyplot"),
		#html.Div(id="training_status",children="my text"),
		
    	
		


        html.H2(
			children="Model testing",
			style={"textAlign": "center"}
		),
		html.Div([
    		html.H5("Upload Test Vector File"),
    		dcc.Upload(
        	id='upload-data',
        	children=html.Div([
            	'Drag and Drop or ',
            	html.A('Select Files')
        	]),

			style={
            	'width': '25%',
            	'height': '60px',
            	'lineHeight': '60px',
            	'borderWidth': '1px',
            	'borderStyle': 'dashed',
            	'borderRadius': '5px',
            	'textAlign': 'center',
            	'margin': '10px'
        	},

        	multiple=False),
    		html.Br(),
			dcc.Store(id='table')
		]),

		html.Div([
    		html.H3('Result of Given Test Vector'),
    		dcc.Graph(id="graph", figure=fig),
    		dcc.Clipboard(target_id="structure"),
    		html.Pre(
        		id='structure',
        		style={
            		'border': 'thin lightgrey solid', 
            		'overflowY': 'scroll',
            		'height': '275px'
        		}
    		),
		]),

	
		
        
        

    ]
   

    
)





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

@app.callback(Output("progressdisplay", "children"),
	Input("trainprogress", "n_intervals"))
	

def update_progressdisplay(n):
	
	return json.dumps(global_data.train_status, indent=4)


@app.callback(Output("historyplot","figure"),
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
	if global_data.model_data["model_type"] == "mlp":
		X_train, X_test, y_train, y_test,n_features= global_data.mlp_training_data()
		model, history=mlp_train(X_train, X_test, y_train, y_test,n_features)
	

	global_data.model_data["model"] = model # keep the trained model
	history = pd.DataFrame(history.history)
	fig = px.line(history, title="Model training metrics")
	#fig.update_layout(xaxis_title="epochs",
	#	yaxis_title="metric value", legend_title="metrics")
	return fig
 



# file upload function
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None
    return df


# callback table creation
@app.callback(Output('table', 'data'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')])

def update_table(contents, filename):
	if contents is not None:
		df = parse_contents(contents, filename)
		if df is not None:
			return df.to_dict()
			


# run server, with hot-reloading
app.run_server(debug=True, threaded=True)
