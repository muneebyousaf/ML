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
	if(file == "2.xlsx") or ((file == "3.xlsx") ) :
		print(file)
		mypd=pd.concat([mypd]*5, ignore_index=True)

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

df1 = df1[['simulaton tick','L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','Last Level Cache Misses','DRAM Page Hit Rate ','status']] 
test_df1=df1[['L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','Last Level Cache Misses','DRAM Page Hit Rate ','status']]

#print(df1)
X, y = df1.values[:,1 :-1], df1.values[:, -1]

test_df1.to_csv("mycsv.csv", sep=',',index=False,header=False)

X = X.astype('float32')
y = LabelEncoder().fit_transform(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(y)
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

exit(0);


total_count=0
wrong_predict=0
true_predict=0
with open('mycsv.csv', newline='') as f:
	reader = csv.reader(f)
	mylist=list(reader)

for i in mylist:
	status=i.pop()
	m=list(np.float_(i))
	yhat= model.predict([m])
	print(status)
	print('Predicted: %.3f' % yhat)
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
		