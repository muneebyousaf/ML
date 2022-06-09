
import glob
import pandas as pd
import random 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# default values
model_data = {
"activation": "relu",
"optimizer": "adam",
"epochs": 100,
"batchsize": 32,
"model_type": "mlp",
}

train_status = {
"running": False,
"epoch": 0,
"batch": 0,
"batch metric": None,
"last epoch": None,

}


def mlp_training_data():
    file_list = glob.glob("*.xlsx")

# list of excel files we want to merge.
# pd.read_excel(file_path) reads the excel
# data into pand
    excl_list = []
    for file in file_list:
        #print(file);
        mypd=pd.read_excel(file)
        print(mypd.shape)
        excl_list.append(mypd)

    # create a new dataframe to store the
    # merged excel file.
    excl_merged = pd.DataFrame()
    for excl_file in excl_list:
	
	# appends the data into the excl_merged
	# dataframe.
        excl_merged = excl_merged.append(excl_file, ignore_index=True)
    df1=excl_merged.sample(frac=1)

    df1 = df1[['simulaton tick','L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','Last Level Cache Misses','DRAM Page Hit Rate ','status','label']] 
    #test_df1=df1[['L1 Instruction Cache Hits', 'L1 Instruction Cache Misses','L1 Data Cache Hits','L1 Data Cache Misses','Last Level Cache Hits','Last Level Cache Misses','DRAM Page Hit Rate ','label']]
    print(df1)
    #print(test_df1)

    #print(df1)
    X, y = df1.values[:,1 :-2], df1.values[:, -2]

    #test_df1.to_csv("mycsv.csv", sep=',',index=False,header=False)

    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y)
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.43)

    #print(y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    n_features = X_train.shape[1]

    #print(n_features)
    return X_train, X_test, y_train, y_test,n_features