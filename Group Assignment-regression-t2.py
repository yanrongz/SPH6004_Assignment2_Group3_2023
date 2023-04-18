# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 04:15:32 2023

@author: Lenovo
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn import svm
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential
from sklearn.metrics import r2_score

data=pd.read_csv('sph_dynamic.csv')
data=data.drop(columns=[x for x in data if round((data[x].isna().sum()/len(data)*100),2) > 60 ])


IDcounts=data.value_counts('stay_id')
IDcounts=pd.DataFrame(IDcounts)

################################### -24,-48 竖直版本 #################################

data['charttime']=pd.to_datetime(data['charttime'],format='%Y/%m/%d %H:%M')
data['first_starttime']=pd.to_datetime(data['first_starttime'],format='%Y/%m/%d %H:%M')
data['last_endtime']=pd.to_datetime(data['last_endtime'],format='%Y/%m/%d %H:%M')
data['icu_intime']=pd.to_datetime(data['icu_intime'],format='%Y/%m/%d %H:%M')
data=data.sort_values(by=['stay_id','charttime'],ascending=[True, False])
data.index=range(data.shape[0])
data['icu_to_vent']=(data['first_starttime']-data['icu_intime']).apply(lambda x:x.total_seconds()/3600)
last_column=data.pop(data.columns[-1])
data.insert(0,last_column.name,last_column)
#data=data.drop(['first_starttime','last_endtime'],axis=1)


one_day=pd.Timedelta(days=1)
data['pre_day']=(data['first_starttime']-one_day)
two_days=pd.Timedelta(days=2)
data['pre_2days']=(data['first_starttime']-two_days)

#筛选出符合2个时间点的所有数据
a=pd.DataFrame()
for i in IDcounts.index:
    patient_records=data[data['stay_id']==i]
    pre_day_record,pre_2days_records,aft_day_records=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    new_records=pd.DataFrame()
    for j in range(patient_records.shape[0]):
        
        if patient_records['charttime'].iloc[j].date()==patient_records['pre_2days'].iloc[j].date():
            pre_2days_records=pd.concat([pre_2days_records,pd.DataFrame(patient_records.iloc[j]).T],axis=0)
        
        if patient_records['charttime'].iloc[j].date()==patient_records['pre_day'].iloc[j].date():
            pre_day_record=pd.concat([pre_day_record,pd.DataFrame(patient_records.iloc[j]).T],axis=0)
                        
    new_records=pd.concat([pre_2days_records.mean(),pre_day_record.mean()],axis=1).T
    new_records['stay_id']=i
    a=pd.concat([a,new_records],axis=0)


#每个病人的多条records平均值填充
b=pd.DataFrame()
for i in IDcounts.index:
    records=a[a['stay_id']==i]
    records.fillna(records.mean(),inplace=True)
    b=pd.concat([b,records],axis=0)
b.index=range(b.shape[0])
#按照插管时间排序，空值以插管时间最相似的病人的值代替
b=b.sort_values(by=['icu_to_vent','stay_id'],ascending=[True,True])
c=b.iloc[0:b.shape[0],:]
c=c.fillna(method='ffill').fillna(method='bfill')
c=c.drop(['icu_to_vent'],axis=1)
#c.to_csv('timesteps=2_NeuralNetwork_new.csv')



################################### -24,-48 横向版本 #################################
newIDcounts=c.value_counts('stay_id')
newIDcounts=pd.DataFrame(newIDcounts)

d=pd.DataFrame(index=range(newIDcounts.shape[0]),columns=['vent_duration','stay_id','calcium_48','creatinine_48','glucose_48','sodium_48','chloride_48','hemoglobin_48','wbc_48','alt_48','ast_48','alp_48','bilirubin_total_48','pt_48',
                                                                                 'calcium_24','creatinine_24','glucose_24','sodium_24','chloride_24','hemoglobin_24','wbc_24','alt_24','ast_24','alp_24','bilirubin_total_24','pt_24'])

idx=0
for i in newIDcounts.index:
    records=c[c['stay_id']==i]
    d['stay_id'][idx]=i
    d['vent_duration'][idx]=records['vent_duration'].iloc[0]
    d.iloc[idx,2:14]=records.iloc[0,2:]
    d.iloc[idx,14:26]=records.iloc[1,2:]
    idx+=1
#d.to_csv('timesteps=2_ML_new.csv')

d=d.astype('float')
label=d['vent_duration']

data=d.drop(['stay_id','vent_duration'],axis=1)

# scaler=preprocessing.StandardScaler()
# data=scaler.fit_transform(data)
# data=pd.DataFrame(data)
# corr_matrix=data.corr().abs()
# upper_triangle=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
# to_drop=[column for column in upper_triangle.columns if any(upper_triangle[column]>0.80)]
# data_select1=data.drop(to_drop, axis=1)
# selector=VarianceThreshold(0)
# selector.fit(data_select1)
# selected_features=selector.get_support(indices=True)
# selected_column_names=data_select1.columns[selected_features].tolist()
# data_select2=data_select1[selected_column_names]
# x_train,x_test,y_train,y_test=train_test_split(data_select2,label,test_size=0.2,shuffle=True,random_state=5) 

x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.2,shuffle=True,random_state=5) 

def rmse(y_pred,y_actual):
    return np.sqrt(np.mean((y_pred-y_actual)**2))
def mse(y_pred,y_actual):
    return np.mean((y_actual-y_pred)**2)
def mae(y_pred,y_actual):
    return np.mean(np.abs(y_actual-y_pred))

#####################################################
SVM=svm.SVR(kernel='rbf')
param_grid_SVM={'C':[1,2,4,8,16,32,64,128,256,512],
                'gamma':[0.1,0.01,0.001,0.0001]}
grid_search=GridSearchCV(SVM, param_grid=param_grid_SVM, cv=5)
grid_search.fit(x_train,y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

model=svm.SVR(kernel='rbf',C=64,gamma=0.1)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('SVR')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))


###
etr=ExtraTreesRegressor(random_state=66)
param_grid_etr={'n_estimators':[50,100,150,200],
                'max_depth': [5,10,15,20,None],
                'min_samples_split': [2,4,6,8,10],
                'min_samples_leaf': [1,2,4,6,8,10]}
grid_search=GridSearchCV(etr, param_grid=param_grid_etr, cv=5)
grid_search.fit(x_train,y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

model=ExtraTreesRegressor(max_depth=10,min_samples_split=2,min_samples_leaf=10,n_estimators=200,random_state=66)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Extra Tree')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))

###
CB=CatBoostRegressor(random_state=66)
param_grid_CB={'learning_rate':[0.01,0.05],
               'border_count':[80,160,240],
               'iterations':[1000,1500],
               'l2_leaf_reg':[2,4,6]}
grid_search=GridSearchCV(CB, param_grid=param_grid_CB, cv=5)
grid_search.fit(x_train,y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
model=CatBoostRegressor(learning_rate=0.01,border_count=240,iterations=1000,l2_leaf_reg=4,random_state=66)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('CatBoost')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))

###
lgbm=lgb.LGBMRegressor(random_state=66)
param_grid_lgb={
    'learning_rate':[0.1,0.2],
    'n_estimators':[100,120,140,160,180,200],
    'max_depth':[3,4,5,6,7,8],
    'num_leaves':[30,50,70,90],
    'lambda_l1':[0.01,0.05,0.1],
}
grid_search=GridSearchCV(lgbm,param_grid=param_grid_lgb,cv=5)
grid_search.fit(x_train,y_train)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

model=lgb.LGBMRegressor(learning_rate=0.1,n_estimators=100,max_depth=3,num_leaves=30,lambda_l1=0.01,random_state=66)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Lightgbm')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))


###
XG=xgb.XGBRegressor(random_state=66)
param_grid_xgb={'n_estimators':[100,150,200],
                'max_depth':[4,6,8,10],
                'learning_rate':[0.05,0.1,0.2],
                'colsample_bytree':[0.5,0.7,0.9],
                'min_child_weight':[1,3,5]
}
grid_search=GridSearchCV(XG,param_grid=param_grid_xgb,cv=5)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

model=xgb.XGBRegressor(learning_rate=0.05,n_estimators=100,max_depth=4,min_child_weight=5,colsample_bytree=0.5,random_state=66)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Xgboost')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))


###
rf=RandomForestRegressor(random_state=66)
param_grid_rf={'n_estimators':[100,150,200],
               'max_depth':[5,10,15,20,None],
               'min_samples_split':[2,4,6,8,10],
               'min_samples_leaf':[1,2,4,6,8,10]
}
grid_search=GridSearchCV(rf,param_grid=param_grid_rf,cv=5)
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

model=RandomForestRegressor(random_state=66,max_depth=5,min_samples_leaf=10,min_samples_split=2,n_estimators=100)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('Random Forest')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))

########################################################################
from keras.layers import SimpleRNN, Dense
from keras.models import Sequential
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx,test_idx=next(gss.split(c,groups=c['stay_id']))

train_data=c.iloc[train_idx] # data
train_label=train_data['vent_duration'] # label
train_label.index=range(train_label.shape[0])
y_train=np.empty((train_label.shape[0]//2,1)) #label
for i in range(y_train.shape[0]):
    y_train[i]=train_label.loc[i*2]

x_train=train_data.drop(['stay_id','vent_duration'],axis=1) # data
scaler=preprocessing.StandardScaler()
x_train=scaler.fit_transform(x_train)
x_train=x_train.reshape(-1,2,12)


#
test_data=c.iloc[test_idx] # data
test_label=test_data['vent_duration'] # label
test_label.index=range(test_label.shape[0])
y_test=np.empty((test_label.shape[0]//2,1)) #label
for j in range(y_test.shape[0]):
    y_test[j]=test_label.loc[j*2]

x_test=test_data.drop(['stay_id','vent_duration'],axis=1) # data
scaler=preprocessing.StandardScaler()
x_test=scaler.fit_transform(x_test)
x_test=x_test.reshape(-1,2,12)

#RNN
model=Sequential()
#model.add(SimpleRNN(50,input_shape=(2,12)))
model.add(SimpleRNN(512,return_sequences=True,input_shape=(2,12)))
model.add(SimpleRNN(256,return_sequences=True))
model.add(SimpleRNN(128,return_sequences=True))
model.add(SimpleRNN(64,return_sequences=True))
model.add(SimpleRNN(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train, epochs=30, verbose=0)
predictions=model.predict(x_test)

print('RNN')
print('r2=',r2_score(y_test,y_pred))
print('rmse=',rmse(y_pred,y_test))
print('mse=',mse(y_pred,y_test))
print('mae=',mae(y_pred,y_test))