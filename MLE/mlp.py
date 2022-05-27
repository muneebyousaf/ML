# mlp for binary classification
# importing the required modules
from cProfile import label
import glob
import pandas as pd
import csv
import random 
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
	'''
	if(file != "1.xlsx") :
		print(file)
		mypd=pd.concat([mypd]*5, ignore_index=True)

	'''
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

df1 = df1[['simulaton tick','L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','Last Level Cache Misses','DRAM Page Hit Rate ','status','label']] 
test_df1=df1[['L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','Last Level Cache Misses','DRAM Page Hit Rate ','label']]
print(df1)
print(test_df1)

#print(df1)
X, y = df1.values[:,1 :-2], df1.values[:, -2]

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



total_count=0
wrong_predict=0
true_predict=0
with open('mycsv.csv', newline='') as f:
	reader = csv.reader(f)
	mylist=list(reader)



#random_list=random.choices(mylist,k=1000)


mylist.reverse()


rwh_counter=0
rwh_true_prediction=0
rwh_false_prediction=0


FF_counter=0
FF_true_prediction=0
FF_false_prediction=0


SPT_counter=0
SPT_true_prediction=0
SPT_false_prediction=0


PF_counter=0
PF_true_prediction=0
PF_false_prediction=0


FR_counter=0
FR_true_prediction=0
FR_false_prediction=0


ST_counter=0
ST_true_prediction=0
ST_false_prediction=0


SPEC_counter=0
SPEC_true_prediction=0
SPEC_false_prediction=0


for i in mylist:
	print(i)
	status=i.pop()
	m=list(np.float_(i))

	yhat= model.predict([m])
	print(status)
	print('Predicted: %.3f' % yhat)
	total_count += 1

	if(total_count == 1000):
		break
	if( status == 'FR'):
		FR_counter +=1
		if( yhat < 0.3):
			FR_true_prediction +=1
		else:
			FR_false_prediction +=1


	if( status == 'RWH'):
		rwh_counter +=1
		if( yhat < 0.3):
			rwh_true_prediction +=1
			
		else:
			rwh_false_prediction +=1
	


	if( status == 'FF'):
		FF_counter +=1
		if( yhat < 0.3):
			FF_true_prediction +=1
		else:
			FF_false_prediction +=1
	

	if( status == 'SPT'):
		SPT_counter +=1
		if( yhat < 0.3):
			SPT_true_prediction +=1
		else:
			SPT_false_prediction +=1
	
	if( status == 'PF'):
		PF_counter +=1
		if( yhat < 0.3):
			PF_true_prediction +=1	
		else:
			PF_false_prediction +=1
	
	
	if( status == 'ST'):
		ST_counter +=1
		if( yhat < 0.3):
			ST_false_prediction +=1
		else:
			ST_true_prediction +=1

	
	if( status == 'SPEC'):
		SPEC_counter +=1
		if( yhat < 0.3):
			SPEC_false_prediction +=1
		else:
				SPEC_true_prediction +=1

				
print (' Total number of samples: %d' %total_count)
true_predict1= rwh_true_prediction+FF_true_prediction+SPT_true_prediction+PF_true_prediction+FR_true_prediction+ST_true_prediction+SPEC_true_prediction
wrong_predict1= rwh_false_prediction+FF_false_prediction+SPT_false_prediction+PF_false_prediction+FR_false_prediction+ST_false_prediction+SPEC_false_prediction;

print(" true  acc: %d" %true_predict1 )
print(" false auu: %d" %wrong_predict1)
true_percentage= (true_predict1/total_count)* 100;
false_pecentage=(wrong_predict1/total_count)*100;

print( ' percentage of correct prediction: %0.3f' %true_percentage)
print('percentage of wrong prediction: %0.3f ' % false_pecentage)


print('Row hammer sample counter: %d' % rwh_counter)
print(' Number of times row hammer was correctly detected: %d' %rwh_true_prediction )
print(' Number of time row hammer could not be detected: %d '%rwh_false_prediction)


print('Flush-Flush sample counter: %d' % FF_counter)
print(' Number of times Flush-Flush was correctly detected: %d' %FF_true_prediction )
print(' Number of time Flush-Flush could not be detected: %d '%FF_false_prediction)


print('SPECTRE sample counter: %d' % SPT_counter)
print(' Number of times SPECTRE was correctly detected: %d' %SPT_true_prediction )
print(' Number of time SPECTRE could not be detected: %d '%SPT_false_prediction)

print('PreFetch and Flush sample counter: %d' % PF_counter)
print(' Number of times PreFetch and Flush was correctly detected: %d' %PF_true_prediction )
print(' Number of time  PreFetch and Flush could not be detected: %d '%PF_false_prediction)


print('Flush+Reload  sample counter: %d' % FR_counter)
print(' Number of times Flush+Reload  was correctly detected: %d' %FR_true_prediction )
print(' Number of time Flush+Reload  could not be detected: %d '%FR_false_prediction)


print('stream benchmark sample counter: %d' % ST_counter)
print(' Number of times stream benchmark was correctly detected: %d' %ST_true_prediction )
print(' Number of time stream benchmark could not be detected: %d '%ST_false_prediction)


print('SPEC benchmark sample counter: %d' % SPEC_counter)
print(' Number of times SPEC benchmark was correctly detected: %d' %SPEC_true_prediction )
print(' Number of time SPEC benchmark could not be detected: %d '%SPEC_false_prediction)


''''

print(' total test ' +  str(total_count))
print('true predict  ' + str(true_predict))
print('false predict ' + str(wrong_predict))

tt=true_predict/total_count *100
tf1=wrong_predict/total_count*100

print(tt)
print(tf1)
''' 

			
'''

	if( yhat < 0.3):
		predict='RWH'
		predict1='FF'
		predict2='SPT'
		predict3='PF'
		predict4='FR'
	else:
		predict='ST'
		predict1='SPEC'
		
	if(status == predict) or (status == predict1)or  (status == predict2) or  (status == predict3) or  (status == predict4):
		true_predict +=1
	else:
		wrong_predict +=1
		print( "wrong predicted %.3f " %yhat)


print(' total test ' +  str(total_count))
print('true predict  ' + str(true_predict))
print('false predict ' + str(wrong_predict))
	'''	
