# mlp for binary classification
# importing the required modules
from cProfile import label
import glob
import pandas as pd
import csv

import numpy as np
from numpy import asarray
from pandas import read_csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K

# csv files in the path
file_list = glob.glob("*.xlsx")

# list of excel files we want to merge.
# pd.read_excel(file_path) reads the excel
# data into pand
excl_list = []

for file in file_list:

	print(file);
	mypd=pd.read_excel(file)
	if(file == "2.xlsx"):
		mypd=pd.concat([mypd]*9, ignore_index=True)

	print(mypd.shape)
	excl_list.append(mypd)

# create a new dataframe to store the
# merged excel file.
excl_merged = pd.DataFrame()
for excl_file in excl_list:
	
	# appends the data into the excl_merged
	# dataframe.
	excl_merged = excl_merged.append(
	excl_file, ignore_index=True)

df1=excl_merged.sample(frac=1)

df1 = df1[['simulaton tick','L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','DRAM Page Hit Rate ','status']] 
test_df1=df1[['L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','DRAM Page Hit Rate ','status']]

#print(df1)
X, y = df1.values[:,1 :-1], df1.values[:, -1]
print(X)
print(X.shape)
print(y)
X1=X;
test_df1.to_csv("mycsv.csv", sep=',',index=False,header=False)

X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_features = X_train.shape[1]

print(n_features)


# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(" training .....")
print(type(X_train))
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)


loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
print('loss: %.3f' % loss)
'''
X_test = np.asarray(X_test)
for (inin, inout) in  zip(X_test,y_test):
	print(inout)
	print(inin)
	yhat= model.predict([*inin])
	if( yhat < 0.3):
		print(" Row hammer attack detected \n");
	else:
		print(" stream is deted")
'''



RW_row1=[3917, 0 ,2461 , 31 , 0 ,   88.24]	
#RW_row1=np.asarray(RW_row1)

ST_row1=[7745 ,0, 1027 , 3024 ,  3018 , 0.00 ]

RW_row2=[5954 ,0 ,2952, 45, 0, 88.0]
RW_row3=[5954, 0 ,2952, 45 ,0 ,88.0]

RW_row4=[584, 0 ,1268, 87, 0 ,62.79]

ST_row2=[6922 ,0 ,8038 ,0 ,0 ,0.0]
ST_row3=[6855, 0 ,8108, 0 ,0 ,0.0]

#RW_row1=X1[0]
#RW_row1	 = K.constant(RW_row1)
#print(RW_row1)

total_count=0
wrong_predict=0
true_predict=0
with open('mycsv.csv', newline='') as f:
	reader = csv.reader(f)
	mylist=list(reader)

for i in mylist:
	status=i.pop()
	m=list(np.float_(i))
	print(m)
	yhat= model.predict([m])
	total_count += 1
	if( yhat < 0.3):
		predict='RWH'
	else:
		predict='ST'
	if(status == predict):
		true_predict +=1
	else:
		wrong_predict +=1


print(' total test ' +  str(total_count))
print('true predict  ' + str(true_predict))
print('false predict ' + str(wrong_predict))
		

yhat2 = model.predict([RW_row2])
yhat3 = model.predict([RW_row3])
yhat4 = model.predict([RW_row4])
print(' RH Predicted: %.3f' % yhat1)
print(' RH Predicted: %.3f' % yhat2)
print(' RH Predicted: %.3f' % yhat3)
print(' RH Predicted: %.3f' % yhat4)




yhat1 = model.predict([ST_row1])
yhat2 = model.predict([ST_row2])
yhat3 = model.predict([ST_row3])
print(' ST Predicted: %.3f' % yhat1)
print(' ST Predicted: %.3f' % yhat2)
print(' ST Predicted: %.3f' % yhat3)




'''
path = '/home/muneeb/Downloads/ionosphere.csv'
df = read_csv(path, header=None)


#**********************************************************
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
print(X)
#print(df)
# encode strings to integer
y = LabelEncoder().fit_transform(y)
#print(y)
# split into train and test datasets
#print( " before splitting *****************************************")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#print(X_train)
#print(y_train)
# determine the number of input features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
#dot_img_file = '/tmp/model_1.png'
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

print(" training .....")
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# make a prediction
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
b_row=[0,	0	,1	,-1,	0,	0	,0	,0,	1,	1,	1	,-1,	-0.71875,	1,	0,	0	,-1,	1,	1,	1	,-1,	1,	1,	0.5625,	-1,	1,	1	,1	,1	,-1,	1,	1	,1	,1	]
g_row=[1,	0,	1	,-0.00612,	1,	-0.09834	,1	,-0.07649	,1	,-0.10605,	1	,-0.11073,	1,	-0.39489,	1,	-0.15616,	0.92124,	-0.31884,	0.86473,	-0.34534,	0.91693,	-0.44072,	0.9606,	-0.46866,	0.81874	,-0.40372	,0.82681	,-0.42231,	0.75784	,-0.382371,	0.80448	,-0.40575,	0.74354,	-0.45039]


print(len(row))
print(len(b_row))
print(len(g_row))
yhat = model.predict([row])
print(yhat)
print('Predicted: %.3f' % yhat)
yhat = model.predict([b_row])
print(yhat)
print(' brow Predicted: %.3f' % yhat)
yhat = model.predict([g_row])
print(yhat)
print(' grow Predicted: %.3f' % yhat)
'''
