clc
clear all
train_data=load('train_data.mat');
train_data=train_data.data;
test_data=load('test_data.mat');
test_data=test_data.data;

train_data=table2array(train_data);
test_data=table2array(test_data);


FunPara.c1=0.00390625; 
FunPara.c2=0.00390625;
FunPara.c3=128;
FunPara.eps1=0.01;
FunPara.sigma=2;
[train_Time,RMSE]=LSTSVR_PI_func(FunPara,train_data,test_data);
RMSE