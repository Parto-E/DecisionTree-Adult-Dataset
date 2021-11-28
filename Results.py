import pandas as pd
import random
import DesicionTree as dt

def load_data(add):
    data = pd.read_csv(add)
    data.columns = ['Lable','workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    data = pd.DataFrame(data)
    data = data.replace({'<=50K':0})
    data = data.replace({'>50K':1})
    return data

def pre_train(train_data,percentage):
    idx = range(len(train_data))
    idx = list(idx)
    random.shuffle(idx)
    aa = int(percentage*len(train_data)/100)
    idx = idx[:aa]
    x1 = train_data.iloc[idx,:]
    x1.index = range(aa)
    x = x1.iloc[:, 1:]
    y = x1.iloc[:, 0] 
    return x,y

def train_val_split(data,percentage):
    idx = range(len(data))
    idx = list(idx)
    random.shuffle(idx)
    aa = int(percentage*len(data)/100)
    idx1 = idx[:aa]
    idx2 = idx[aa:]
    x_tr = data.iloc[idx1,:]
    x_val = data.iloc[idx2,:]
    x_tr.index = range(aa)
    x_val.index = range(len(data)-aa)
    x_tr_ = x_tr.iloc[:, 1:]
    y_tr_ = x_tr.iloc[:, 0] 
    x_val_ = x_val.iloc[:, 1:]
    y_val_= x_val.iloc[:, 0] 
    
    return x_tr_,y_tr_,x_val_,y_val_ 

data = load_data('adult.train.10k.discrete')
x1,y1,x2,y2 = train_val_split(data,75)
x1,y1 = pre_train(data,100)

test_data=load_data('adult.train.10k.discrete')
x2,y2,x_test,y_test=train_val_split(test_data,100)
x_test = test_data.iloc[:, 1:]
y_test= test_data.iloc[:, 0] 

tree = dt.DecisionTree(max_depth = 6)
Train = tree.fit(x1,y1,x2,y2)
prun = tree.pruning(x2, y2)
Test = tree.evaluation(x_test,y_test)
print(Train)
print(Test)

