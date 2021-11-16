#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:46:45 2020

@author: maliaosaide
"""

import numpy as np
from modin.pandas import DataFrame
from modin.pandas import concat
import matplotlib.pyplot as plt
import modin.pandas as pd
from modin.pandas import read_csv
from matplotlib import pyplot
import math
from modin.pandas import Series
from keras.models import Sequential
from keras.layers import TimeDistributed,Activation
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN,GRU,Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import backend as K
#from keras import evaluate
from matplotlib.font_manager import FontProperties
import os
import warnings

warnings.filterwarnings("ignore")

np.random.seed(7)
#官方F值，Official F value
def evaluate1(predict,real):
    daylen=len(predict)
    daycount=[]
    artcount=[]
    for i in range(daylen):
        count=(float(predict[i]-real[i])/float(real[i]))**2
        bcount=float(real[i])
        artcount.append(bcount)
        daycount.append(count)
        
    a=math.sqrt(  float(sum(daycount))/float(daylen)  )
    b=math.sqrt( float(sum(artcount))      )
    
    f=(1-a)*b
    return f
#时间序列数据切分，Time series data segmentation
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

#单特征文件，Single feature file
def onedata (filename):
    datadict={}
    with open(filename) as f:
        for line in f:
            if "art" in line:
                pass
            else:
                art,date,play=line.split(",")
                if art=="2b7fedeea967becd9408b896de8ff903":
                    continue
                elif art in datadict:
                    datadict[art].append([date,play.split("\n")[0]])
                else:
                    datadict[art]=[[date,play.split("\n")[0]]]
        return datadict

#多特征文件，Multi feature file
def data(filename1):
    datadict={}
    with open(filename1) as f:
        for line in f:
            if "art" in line:
                pass
            else:
                art,date,play,download,collection=line.split(",")
                if art=="2b7fedeea967becd9408b896de8ff903":
                    continue
                elif art in datadict:
                    datadict[art].append([date,play,download,collection.split("\n")[0]])
                else:
                    datadict[art]=[[date,play,download,collection.split("\n")[0]]]
    return datadict


#数据集切分，并进行最大最小归一化
#Data set segmentation. Maximum and minimum normalization
def triantestdata (dictlist,selectday,timestep):#timestep为处理bitch整除用的
    data=dictlist 
    artdata=sorted(data,key=lambda artdata:artdata[0])  
    trainall=artdata[:-30]
    if selectday==0:
        section_data1=artdata[selectday:-30]
    else:
        section_data1=artdata[153-selectday:-30]
    yushu=len(section_data1)%timestep 
    section_data=section_data1[yushu:]
    train = pd.DataFrame(section_data,dtype='float').drop([0], axis=1)
    #7月份数据
    section_data2=artdata[-30:] 
    test = pd.DataFrame(section_data2,dtype='float').drop([0], axis=1)
    return train,test,trainall

#准备训练数据，Prepare training data
def prepare_data(series,columns,timestep):
    
    train = series.values
    train_X= train[:, :columns]  
    train_y =train[:, -columns]
    lasttrain=train[-timestep:,:columns].reshape(1, -1, columns)
    train_X = train_X.reshape(( -1, timestep , train_X.shape[1])) 
    #多特征输出y，Multi feature output y
    train_y1=train_y.reshape((-1,timestep , train_X.shape[2])) 
    #单特征输出y，Single feature output y
    train_y2=train_y.reshape(-1,timestep)
    return train_X,train_y1,train_y2,lasttrain


def maxmin(train,test,timestep=1):
    train=pd.DataFrame(train)
    test=pd.DataFrame(test)
    min_max_scaler = MinMaxScaler() 
    lie=train.shape[1] 
    X_train_minmax = min_max_scaler.fit_transform(train) 
    a=series_to_supervised(X_train_minmax,timestep,1)  
    #a=series_to_supervised(X_train_minmax,2,1) 的输出结果：
    #       var1(t-2)  var2(t-2)  var3(t-2)  ...   var1(t)  var2(t)   var3(t)
    # 2     0.031161     0.0000   0.117188  ...  0.036827   0.1250  0.039062
    # 3     0.000000     0.0000   0.132812  ...  0.031161   0.0000  0.101562
    # 4     0.036827     0.1250   0.039062  ...  0.084986   0.0625  0.039062
    # 5     0.031161     0.0000   0.101562  ...  0.144476   0.0000  0.000000
    # 6     0.084986     0.0625   0.039062  ...  0.127479   0.0000  0.007812
    b=list(a.values[-timestep:,:lie])#和lasttrain值一样
    train_X,train_y,train_y1,lasttrain=prepare_data(a,lie,timestep)
    return train_X,train_y,train_y1,b,lasttrain,min_max_scaler #训练集的最后一个


def OFLSTM (train_X, train_y,test_X,test_y,timestep,feture):
    #单特征LSTM，可循环
    model = Sequential()
    model.add(LSTM(64,return_sequences=True,activation="relu",input_shape=(timestep,feture)))
    model.add(LSTM(32,activation="relu"))
    # model.add(LSTM(feture,))  
    model.add(Dense(timestep))
    # model.add(Dense(timestep))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    history = model.fit(train_X, train_y, 
                            epochs=200, 
                            batch_size=32, 
                            verbose=0, shuffle=False,
                            validation_data=(test_X,test_y),
                        )
    return model,history

def BiLSTM (train_X, train_y,test_X,test_y,timestep,feture):
    #单特征LSTM，可循环
    model = Sequential()
    model.add(Bidirectional(LSTM(64,return_sequences=True,activation="relu",input_shape=(timestep,feture))))
    model.add(Bidirectional(LSTM(32,activation="relu",return_sequences=True)))
    # model.add(LSTM(2,activation="relu",return_sequences=True))
    model.add(Dense(timestep))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    history = model.fit(train_X, train_y, 
                            epochs=200, 
                            batch_size=32, 
                            verbose=0, shuffle=False,
                            validation_data=(test_X,test_y),
                        )
    return model,history


def myGRU (train_X, train_y,est_tX,test_y,timestep,feture):
    #单特征LSTM，可循环
    model = Sequential()
    model.add(GRU(64,return_sequences=True,activation="relu",input_shape=(timestep,feture)))
    model.add(GRU(32,activation="relu"))
    # model.add(LSTM(2,activation="relu",return_sequences=True))
    model.add(Dense(timestep))
    model.add(Activation("tanh"))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    history = model.fit(train_X, train_y, 
                            epochs=200, 
                            batch_size=32, 
                            verbose=0, shuffle=False,
                            validation_data=(test_X,test_y),
                        )
    return model,history

#RPA算法，滚动预测算法，Rolling prediction algorithm
def timeperdict(model,last_train,rollstep,timestep):
    predict_trian=[last_train.reshape(1,-1)] #这个里面是7月最后两天的数据，Data for the last two days of July
    trainPredict_list=[]
    r=rollstep
    for i in range(0,30,1):
        trainPredict = model.predict(last_train) 
        trainPredict_list.append(trainPredict)  
        predict_trian.append(trainPredict)
        
        a=np.array(predict_trian).reshape(1,-1)
        b=a[:,r:r+timestep]
        
        c=b.reshape(1,timestep,last_train.shape[2])  #变成模型输入格式，Change to model input format
        trainPredict=c  
        
        last_train=trainPredict
        r=r+rollstep  #滚动向前

    return trainPredict_list





# filename="1.csv" #单特征数据
# data=onedata(filename)
# filename="aaa.csv" #多特征数据
# data=data(filename)
#超参数Super parameter
dataset=0 #训练集大小，0默认全部
timestep=1   #时间步长
featurelie=1   #特征列
rollstep=1   #滚动步长
dirname="2时间2滚动gru加入Activation函数tanh"

#选定最多8个时间步长
for i in range(1,9):
    # timestep=1
    for j in range(1,9):
        timestep=j
        dirname=str(timestep)+"时间"+str(i)+"滚动,复现之前的实验"
        if timestep<rollstep:
            continue
        else:
            print("canshu")
            print(timestep,rollstep)
            all_perdict_sorce=[]
            all_sorce=[]
            
                
            if  os.path.exists("picture/"+dirname):
                print("文件夹已存在")
            else:
                
                os.makedirs("picture/"+dirname)
                
                for art in data:
                    artdata=data[art]
                    
                    train,test,section_data=triantestdata(artdata,dataset,timestep)
                    
                    x,y,y1,a,lasttrain,min_max_scaler=maxmin(train,test,timestep)
                    
                    batch1=x.shape[0]
                    num=int(batch1*0.8)
                    num1=int(batch1*0.2)
                    train_X=x[-num:]
                    train_y=y1[-num:]
                    test_X=x[-num1:]
                    test_y=y1[-num1:]
                    
                    model,history=OFLSTM(train_X, train_y,test_X,test_y,timestep,featurelie)
                    perdict=timeperdict(model,lasttrain,rollstep,timestep)
                    
                    # model,history=BiLSTM(train_X, train_y,test_X,test_y,timestep,featurelie)
                    # perdict=timeperdict(model,lasttrain,rollstep,timestep)
                    
                    # model,history=myGRU(train_X, train_y,test_X,test_y,timestep,featurelie)
                    # perdict=timeperdict(model,lasttrain,rollstep,timestep)
                    
                    Predict_array=min_max_scaler.inverse_transform(np.array(perdict).reshape(1,-1))[:,:30] #取最前的30天
                    real_list=test.values
                    
                    # Predict_array=[i for i in Predict_array[0] ]
                    # real_list=[i for i in real_list[0] ]
                    
                    
                    e=evaluate1(Predict_array.T,real_list)
                    ee=evaluate1(real_list,real_list)
                    all_perdict_sorce.append(e)
                    all_sorce.append(ee)
                    x=[i for i in range(30)]
                        
                    plt.plot(x,Predict_array.T,label="$predict$",color="blue")
                    plt.plot(x,real_list,label="$real$",color="orange") 
                        
                    font =FontProperties(fname='PingFang.ttc') 
                    plt.legend(loc = 'upper right')

                    plt.ylabel("播放量",fontproperties=font)
                    plt.xlabel("天数",fontproperties=font)
                     
                    plt.savefig("picture/"+dirname+"/"+art+" "+str(e)+" "+str((e/ee)*100)+".svg")
                        
                    plt.show()
                    
                    
                    K.clear_session()
                
            print("全部")
            a=sum(all_perdict_sorce)
            b=sum(all_sorce)
            print(len(all_perdict_sorce),len(all_sorce))
            print(a,b,(a/b)*100)
            
            with open("picture/"+dirname+"/result.txt","w")as f:
                f.write("总分："+str(a)+"\n")
                f.write("满分："+str(b)+"\n")
                f.write("占比："+str((a/b)*100)+"\n")
        
            
            
    rollstep+=1   
    
   

    
    
    










































