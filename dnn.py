import pandas as pd
import numpy as np
import random as rn
import tensorflow as tf
import xgboost as xgb

from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from keras import backend as K
import keras

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from lightgbm import LGBMClassifier

from xgboost import plot_importance

import matplotlib.pyplot as plt

#데이터 분석 함수
def data_analysis(creditcard, X_train, X_val, X_test, y_train):
  print(X_train.shape)
  print("1의 비율 : ", sum(y_train['Class'])/X_val.shape[0] )
  print("0의 비율 : ", (X_val.shape[0]-sum(y_train['Class'])) /X_val.shape[0] )

  #데이터 확인
  print(creditcard.info()) # All Non-Null 
  print(creditcard.describe()) #mean, std, 4분위수 값들
  #Time과 Amount 변수 box plot 그려보기
  plt.figure(figsize=(7,5))
  plt.subplot(121)
  plt.boxplot(creditcard['Time'], labels = ['Time'],showmeans=True)
  plt.subplot(122)
  plt.boxplot(creditcard['Amount'], labels = ['Amount'], showmeans=True)
  plt.show()

# 학습 과정의 loss를 plotting 하는 함수
def loss_plotting(history):

    history_dict = history.history

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1,len(loss)+1)

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'bo', label = 'Validation Loss')

    plt.title('Training and validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def data_preprocessing(X_train, y_train, X_val, X_test, feature):
  '''
    ##전처리 함수 data_preprocessing :이상치 삭제,변환 및 표준화 (유진) 
    #input: X_train, y_train, X_val, X_test, feature(= 이상치 처리할 변수 ex.'V15')
    #output : Xtrain_scaled, ytrain_scaled, Xval_scaled, Xtest_scaled
  ''' 

  ytrain_scaled = y_train.copy() #원본 데이터 변경 없이 copy()하여 전처리 진행
  Xtrain_scaled = X_train.copy()
  Xval_scaled = X_val.copy()
  Xtest_scaled = X_test.copy()

  ### 표준화(mean=0, std=1)
  # train 데이터로만 fit하고 train&test에 적용
  scaler = StandardScaler()
  for x in ["Amount","Time"]: #"Amount","Time"변수 표준화 진행
    Xtrain_scaled[x] = scaler.fit_transform(Xtrain_scaled[x].values.reshape(-1,1))
    Xtest_scaled[x] = scaler.transform(Xtest_scaled[x].values.reshape(-1,1))
    Xval_scaled[x] = scaler.transform(Xval_scaled[x].values.reshape(-1,1))
  
  # ndarray -> DataFrame으로 바꾸기(StandardScaler 수행 결과, numpy nadarray로 바뀌기에 변환 과정 필요)
  Xtrain_scaled = pd.DataFrame(data=Xtrain_scaled , index=X_train.index, columns=X_train.columns)
  Xtest_scaled = pd.DataFrame(data=Xtest_scaled, index=X_test.index, columns=X_test.columns)
  Xval_scaled = pd.DataFrame(data=Xval_scaled, index=X_val.index, columns=X_val.columns)
  
  ### 이상치 처리
  # 1) train data : class 0의 outlier 삭제
  ytrain_0 = ytrain_scaled.loc[ytrain_scaled['Class'] == 0] 
  ytrain_0_idx = [idx for idx in ytrain_0.index] #train data 중 class가 0인 데이터의 index를 list로 추출
  Xtrain_scaled_0 = Xtrain_scaled.loc[ytrain_0_idx, :] #train data중 class가 0인 데이터 추출
  
  qt_25 = np.percentile(Xtrain_scaled_0[feature].values, 25)
  qt_75 = np.percentile(Xtrain_scaled_0[feature].values, 75)

  IQR = qt_75 - qt_25
  IQR_weight = IQR*1.5

  lowest = qt_25 - IQR_weight
  highest = qt_75 + IQR_weight

  # outlier idx 찾기
  outlier_idx = Xtrain_scaled_0[feature][ (Xtrain_scaled_0[feature] < lowest) | (Xtrain_scaled_0[feature] > highest) ].index
  Xtrain_scaled.drop(outlier_idx, axis=0, inplace=True) #Xtrain 이상치 삭제(drop)
  ytrain_scaled.drop(outlier_idx,  axis=0, inplace=True) #ytrain 이상치 삭제(drop)
  print("train 데이터에서 삭제한 이상치 개수 : ",len(outlier_idx))
  print("train 데이터 이상치 삭제 shape : ", Xtrain_scaled.shape)

  # 2) validation, test : 이상치 데이터 삭제 대신 np.clip 사용하여 이상치 min,max값으로 치환
  #validation data
  qt_25 = np.percentile(Xval_scaled[feature].values, 25)
  qt_75 = np.percentile(Xval_scaled[feature].values, 75)
  IQR = qt_75 - qt_25
  IQR_weight = IQR*1.5
  lowest = qt_25 - IQR_weight
  highest = qt_75 + IQR_weight
  Xval_scaled[feature] = np.clip(Xval_scaled[feature],lowest,highest) 

  #test data
  qt_25 = np.percentile(Xtest_scaled[feature].values, 25)
  qt_75 = np.percentile(Xtest_scaled[feature].values, 75)
  IQR = qt_75 - qt_25
  IQR_weight = IQR*1.5
  lowest = qt_25 - IQR_weight
  highest = qt_75 + IQR_weight
  Xtest_scaled[feature] = np.clip(Xtest_scaled[feature],lowest,highest)

  return Xtrain_scaled, ytrain_scaled, Xval_scaled, Xtest_scaled

def compare_scaler(X_train, X_val, X_test,type):
    '''
    # Scaling 방법 비교 함수
    # Input : X_train, X_val, X_test, type(어떤 scaling방법을 이용할지 선택, 1~4)
    # Output : Xtrain_scaled, Xval_scaled, Xtest_scaled
    '''

    # 원본 데이터 copy
    Xtrain_scaled = X_train.copy()
    Xval_scaled = X_val.copy()
    Xtest_scaled = X_test.copy()
    feature = X_train.columns[:]

    # 모든 feature에 대해서 이상치 수정(iqr 방식 이용)
    for i in feature:
        # X_train 데이터 이상치 수정
        quantile_25 = np.percentile(Xtrain_scaled[i].values, 25)
        quantile_75 = np.percentile(Xtrain_scaled[i].values, 75)
        IQR = quantile_75 - quantile_25
        IQR_weight = IQR*1.5
        lowest = quantile_25 - IQR_weight
        highest = quantile_75 + IQR_weight
        Xtrain_scaled[i] = np.clip(Xtrain_scaled[i],lowest,highest)
        
        # X_val 데이터 이상치 수정
        quantile_25 = np.percentile(Xval_scaled[i].values, 25)
        quantile_75 = np.percentile(Xval_scaled[i].values, 75)
        IQR = quantile_75 - quantile_25
        IQR_weight = IQR*1.5
        lowest = quantile_25 - IQR_weight
        highest = quantile_75 + IQR_weight
        Xval_scaled[i] = np.clip(Xval_scaled[i],lowest,highest)
        
        # X_test 데이터 이상치 수정
        quantile_25 = np.percentile(Xtest_scaled[i].values, 25)
        quantile_75 = np.percentile(Xtest_scaled[i].values, 75)
        IQR = quantile_75 - quantile_25
        IQR_weight = IQR*1.5
        lowest = quantile_25 - IQR_weight
        highest = quantile_75 + IQR_weight
        Xtest_scaled[i] = np.clip(Xtest_scaled[i],lowest,highest)

    # type 값에 따라 다른 scaling 방식 적용
    if type == 1:                                       #1) 표준화(mean=0, std=1)
        scaler = StandardScaler()
        Xtrain_scaled = scaler.fit_transform(Xtrain_scaled)
        Xtest_scaled = scaler.transform(Xtest_scaled)
        Xval_scaled = scaler.transform(Xval_scaled)
    elif type == 2:                                     #2) 정규화(0~1사이 값)
        scaler = MinMaxScaler()
        scaler.fit(Xtrain_scaled)
        Xtrain_scaled = scaler.transform(Xtrain_scaled)
        Xval_scaled = scaler.transform(Xval_scaled)
        Xtest_scaled = scaler.transform(Xtest_scaled)
    elif type == 3:                                     #3)정규화와 비슷(-1~1사이 값)
        scaler = MaxAbsScaler()
        scaler.fit(Xtrain_scaled)
        Xtrain_scaled = scaler.transform(Xtrain_scaled)
        Xval_scaled = scaler.transform(Xval_scaled)
        Xtest_scaled = scaler.transform(Xtest_scaled)
    elif type == 4:                                     #4)IQR 사용(이상치 제거하지 않은 값 사용)
        scaler = RobustScaler()
        scaler.fit(X_train)
        Xtrain_scaled = scaler.transform(X_train)
        Xval_scaled = scaler.transform(X_val)
        Xtest_scaled = scaler.transform(X_test)

    return Xtrain_scaled, Xval_scaled, Xtest_scaled

def change_data(Xtrain_scaled, y_train):
  '''
  ###OverSampling, Undersampling 데이터 만드는 함수 : change_data
  #input : Xtrain_scaled, y_train
  #output : Xtrain_smote, ytrain_smote, Xtrain_adasyn, ytrain_adasyn, Xtrain_under, ytrain_under
  '''  
  #1) SMOTE
  Xtrain_smote, ytrain_smote = SMOTE(random_state=0).fit_sample( Xtrain_scaled, y_train['Class'] )
  sns.countplot(x = ytrain_smote, hue =ytrain_smote) #sns plot사용하여 클래스 비율 시각화해보기
  plt.show()
  print("After OverSampling (SMOTE) | Xtrain_smote shape : ", Xtrain_smote.shape)
  #2) ADASYN
  Xtrain_adasyn, ytrain_adasyn = ADASYN(random_state=0).fit_sample( Xtrain_scaled, y_train['Class'] )

  #3) UnderSampling
  rus = RandomUnderSampler(random_state=0)
  Xtrain_under, ytrain_under = rus.fit_resample(Xtrain_scaled, y_train['Class'])
  sns.countplot(x = ytrain_under, hue =ytrain_under)
  plt.show()
  print("After UnderSampling | Xtrain_under shape : ",Xtrain_under.shape)
  return Xtrain_smote, ytrain_smote, Xtrain_adasyn, ytrain_adasyn, Xtrain_under, ytrain_under

# Metrics 리스트 (accuracy, precision, recall, f1_score, auc)
#Metrics 함수 출처 : https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#F1 Score
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
Metrics= [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      f1_m,
      keras.metrics.AUC(name='auc'),
]

# DNN 모델 생성 함수(30,15,15,15,1의 unit 수를 갖는 모델) 
def dnn_model(Xtrain=None, metrics = Metrics, optimizer = None):
  model = keras.Sequential([
      keras.layers.Dense(15, activation='relu',input_shape=(Xtrain.shape[-1],)),
      keras.layers.Dense(15, activation='relu'),
      keras.layers.Dense(15, activation='relu'),
      keras.layers.Dense(1, activation='sigmoid'),
  ])
  model.compile(optimizer=optimizer,
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)
  return model

# Stacking Ensemble DNN 모델 생성 함수
def dnn_model2(Xtrain=None, metrics = Metrics, optimizer = None):
  model = keras.Sequential([
      keras.layers.Dense(15, activation='relu',input_shape=(Xtrain.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid'),
  ])
  model.compile(optimizer=optimizer,
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)
  return model

####Optimizers (sgd vs adagrad vs adam vs rmsprop) 별 성능 비교 위한 실험 함수 : find_optimizer
# input : X_train, y_train, X_val, X_test
# output : 실험 결과 'optimizers_df' dataframe
  
#결과값 시각화하기하기 위한 함수
### loss 그래프 ##
def optimizers_loss(optimizers_dict, TrainLoss_dict): 
  Epochs = 7
  markers = {"SGD": "o", "AdaGrad": "s", "Adam": "D", "RMSProp":"*"}
  x = np.arange(Epochs)
  plt.figure(figsize=(20,8))
  for key in optimizers_dict.keys():
      plt.plot(x, TrainLoss_dict[key], marker=markers[key], markevery=100, label=key)
  plt.xlabel("iterations")
  plt.ylabel("loss")
  plt.ylim(0, 1)
  plt.legend()
  plt.show()

### roc 곡선 ## 

def optimizers_roc(optimizers_dict, Roc_dict):
  markers = {"SGD": "o", "AdaGrad": "s", "Adam": "D", "RMSProp":"*"}
  plt.figure(figsize=(20,8))
  for key in optimizers_dict.keys():
    fpr = Roc_dict[key][0]
    tpr = Roc_dict[key][1]
    plt.plot(100*fpr, 100*tpr,  marker=markers[key], label=key)
  plt.xlabel('False positives')
  plt.ylabel('True positives')
  plt.xlim([-0.5,100.5])
  plt.ylim([-0.50,100.5])
  plt.grid(True)
  plt.legend()
  plt.show()

def find_optimizer(X_train, y_train, X_val, y_val, X_test,y_test):

  #optimizer 선언
  optimizers_dict = {}
  optimizers_dict['SGD']=SGD(lr=0.01, momentum=0.9)
  optimizers_dict['AdaGrad']=Adagrad(lr=0.01)
  optimizers_dict['Adam']=Adam()
  optimizers_dict['RMSProp']=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
  
  #loss그래프, ROC 시각화하기 위함
  TrainLoss_dict = {}
  TestLoss_dict = {} 
  Roc_dict = {}

  Epochs = 15
  BatchSize = 2048

  #데이터 전처리
  Xtrain_scaled, ytrain_scaled, Xval_scaled, Xtest_scaled = data_preprocessing(X_train, y_train, X_val, X_test,'V15') 

  #optimizer별 성능 결과 기록 위한 dataframe
  optimizers_df = pd.DataFrame(columns =['optimizer','loss', 'accuracy','precision', 'recall','f1','auc'])

  #optimizer 별 학습
  for key in optimizers_dict.keys():
    optimizer_list = []
    optimizer_list.append(key)

    #모델생성
    model = dnn_model(Xtrain = Xtrain_scaled, optimizer = key) 
    #model.summary()
    #모델학습
    hist = model.fit(Xtrain_scaled, ytrain_scaled, batch_size = BatchSize, epochs = Epochs,
            validation_data=(Xval_scaled,y_val))

    #모델평가
    results = model.evaluate(Xtest_scaled, y_test,batch_size=BatchSize) 
    for name, value in zip(model.metrics_names, results):
      print(name, ': ', value)
      optimizer_list.append(value)   
    optimizers_df.loc[len(optimizers_df)] = optimizer_list

    #예측
    ytest_pred = model.predict(Xtest_scaled).ravel()
    fpr, tpr, _ = roc_curve(y_test, ytest_pred)
    Roc_dict[key]=(fpr, tpr) #(fp, tp)튜플로 저장
    
    # 학습과정 loss 살펴보기
    print('##{} : training loss and acc ##'.format(key))
    print("\nHistory loss : ", hist.history['loss'])
    TrainLoss_dict[key] = hist.history['loss']
    print("\nHistory ACC : ", hist.history['accuracy'])
    TestLoss_dict[key] = hist.history['loss']
  
  # optimizers_df 결과 확인
  print(optimizers_df)
  # 시각화 결과 확인
  optimizers_loss(optimizers_dict,TrainLoss_dict) #학습 loss 그리기 위해 호출
  optimizers_roc(optimizers_dict, Roc_dict) #ROC곡선 그리기 위해 호출
  return optimizers_df
  ################################

def find_classweight(X_train, y_train, X_val,y_val, X_test,y_test):
  '''
  ####Class Weight 황금 비율 찾는 함수 : find_classweight
  #input : X_train, y_train, X_val, X_test
  #output : 실험 결과 'weight_df' dataframe
  '''
  Epochs = 15
  BatchSize = 2048

  #데이터 전처리
  Xtrain_scaled, ytrain_scaled, Xval_scaled, Xtest_scaled = data_preprocessing(X_train, y_train, X_val, X_test,'V15')

  #weight별 성능 확인할 dataframe
  weight_df = pd.DataFrame(columns =['weight0','loss', 'accuracy','precision', 'recall','f1','auc'])
  ##################################
  #class weight 적용 X
  weight_list = []
  weight_list.append(0)
  model = dnn_model(Xtrain = Xtrain_scaled, optimizer=RMSprop())
  hist = model.fit(Xtrain_scaled, ytrain_scaled, batch_size = BatchSize, epochs = Epochs, 
            validation_data=(Xval_scaled,y_val))
  #평가
  results = model.evaluate(Xtest_scaled, y_test,batch_size=BatchSize)
  for name, value in zip(model.metrics_names, results):
    print(name, ': ', value)
    weight_list.append(value)   
  weight_df.loc[len(weight_df)] = weight_list #weight_df에 추가하기

  ####################################
  #class weight 적용 -> grid search로 최적의 값 찾기
  for w in np.arange(0.01, 0.20, 0.01):
    class_weight = {0: w, 1: 1-w} #class weight 지정
    model1 = dnn_model(Xtrain = Xtrain_scaled, optimizer=RMSprop())
    hist1 = model1.fit(Xtrain_scaled, ytrain_scaled, batch_size = BatchSize, epochs = Epochs, 
              validation_data=(Xval_scaled,y_val), class_weight = class_weight) # 학습 시 class weight반영 

    #평가
    results1 = model1.evaluate(Xtest_scaled, y_test,batch_size=BatchSize)
    weight_list = []
    weight_list.append(w)
    for name, value in zip(model1.metrics_names, results1):
      print(name, ': ', value)   
      weight_list.append(value)
    weight_df.loc[len(weight_df)] = weight_list #weight_df에 추가하기

  print(weight_df)
  return weight_df

#weight_df = find_classweight(X_train, y_train, X_val,y_val, X_test,y_test)

######################################################3
###스태킹 앙상블
#스태킹 구현 함수  :stacking_ensemble
#input : 학습 및 평가할 데이터 (X_train, y_train, X_val, X_test)
#output : 최종 결과 (acc, precision, recall, fscore,auc) print
#스태킹 과정 중 필요한 함수 : stacking_model 
#각 모델별로 진행 
def stacking_model(model, Xtrain_scaled, ytrain_scaled, Xtest_scaled, k=5):
  train_fold_pred = np.zeros((Xtrain_scaled.shape[0], 1))
  test_fold_pred = np.zeros((Xtest_scaled.shape[0], k))
  stk = StratifiedKFold(n_splits=k)

  # k fold 로 나누기 -> for문으로 k번 진행
  for i, (train_idx, valid_idx) in enumerate(stk.split(Xtrain_scaled,ytrain_scaled)):
    #print(train_idx)
    Xtrain_k = Xtrain_scaled.iloc[train_idx] #kfold에서 학습용
    ytrain_k = ytrain_scaled.iloc[train_idx]
    Xval_k = Xtrain_scaled.iloc[valid_idx] #kfold에서 검증용
    
    # 개별 모델들 학습. 
    model.fit(Xtrain_scaled,ytrain_scaled)
    
    # 검증데이터로 검증 & 예측값 train_fold_predict에 차곡차곡 저장
    train_fold_pred[valid_idx, :] = model.predict(Xval_k).reshape(-1, 1)
    
    #해당 폴드에서 생성된 모델로 Xtest_scaled 예측 & 저장
    test_fold_pred[:, i] = model.predict(Xtest_scaled)
    
  #test_fold_pred
  test_pred_mean = np.mean(test_fold_pred, axis =1).reshape(-1, 1)
    
  return train_fold_pred, test_pred_mean

def stacking_ensemble(X_train, y_train, X_val, X_test, y_test):

  Epochs = 20
  BatchSize = 2048

  #데이터 전처리
  Xtrain_scaled, ytrain_scaled, Xval_scaled, Xtest_scaled = data_preprocessing(X_train, y_train, X_val, X_test,'V15')
  
  #모델 학습 gbc, rf, dt, ada
  gbc = GradientBoostingClassifier(random_state=0) 
  rf = RandomForestClassifier(n_estimators=100, random_state=0)
  dt = DecisionTreeClassifier()
  ada = AdaBoostClassifier(n_estimators=100)
  
  #Step1 개별 모델 stacking할 데이터 얻기 (gbc, rf, dt, ada)
  gbc_train_lv2, gbc_test_lv2 = stacking_model(gbc, Xtrain_scaled, ytrain_scaled, Xtest_scaled)
  rf_train_lv2, rf_test_lv2 = stacking_model(rf, Xtrain_scaled, ytrain_scaled, Xtest_scaled)
  dt_train_lv2, dt_test_lv2 = stacking_model(dt, Xtrain_scaled, ytrain_scaled, Xtest_scaled)
  ada_train_lv2, ada_test_lv2 = stacking_model(ada, Xtrain_scaled, ytrain_scaled, Xtest_scaled)

  #Step2 최종 모델에 사용할 데이터로 합치기 
  Xtrain_lv2 = np.concatenate((gbc_train_lv2, rf_train_lv2, dt_train_lv2,ada_train_lv2), axis = 1)
  Xtest_lv2 = np.concatenate((gbc_test_lv2, rf_test_lv2,dt_test_lv2, ada_test_lv2), axis = 1)

  print("Original data : ", Xtrain_scaled.shape, Xtest_scaled.shape)
  print("New lv2 data : ", Xtrain_lv2.shape, Xtest_lv2.shape)

  lv2_model = dnn_model2(Xtrain = Xtrain_lv2, optimizer=RMSprop())
  #model.summary()
  hist = lv2_model.fit(Xtrain_lv2, ytrain_scaled, batch_size = BatchSize, epochs = Epochs) 
            #validation_data=(Xval_scaled,y_val))

  #평가
  # StackingEnsemble 모델 평가
  print("Stacking Ensemble")
  results = lv2_model.evaluate(Xtest_lv2, y_test,batch_size=BatchSize)
  for name, value in zip(lv2_model.metrics_names, results):
      print(name, ': ', value)   

#stacking_ensemble(X_train, y_train, X_val, X_test, y_test) #스태킹 앙상블 실행

###################
# Voting classifier model 생성 함수
def voting_model(model,X_train,y_train) :
    estimator = []
    index=0
    for clf in model :
        estimator.append(('clf'+str(index),clf))
        index = index + 1
    vo_clf = VotingClassifier(estimators=estimator,voting='soft')

    # estimators에 입력한 모델 각각 train 데이터로 학습
    vo_clf.fit(X_train,y_train.values.ravel())
    return vo_clf

# Voting classifier 테스트 함수
def voting_classifier(X_train, y_train, X_test, y_test):
    Epochs = 15
    BatchSize = 2048

    # Voting에 사용할 classifier 생성 (logistic, KNN, DecisionTree, SVM, DNN)
    lr_clf = LogisticRegression()
    knn_clf = KNeighborsClassifier(n_neighbors = 3)
    dt_clf = DecisionTreeClassifier()
    svm_clf = SVC(probability=True)
    dnn_clf = KerasClassifier(build_fn = lambda: dnn_model(Xtrain=X_train, optimizer=RMSprop(learning_rate=1e-3)), epochs = Epochs, batch_size=BatchSize)
    dnn_clf._estimator_type="classifier"

    # Voting에 사용할 classifier 모음 (두번의 테스트)
    classifiers = [lr_clf,knn_clf,dt_clf,svm_clf,dnn_clf]
    clf = [dnn_clf,dnn_clf,dnn_clf,dnn_clf,dnn_clf]

    #Voting classifier model 생성
    vo_clf = voting_model(classifiers,X_train,y_train)

    # model 평가
    pred = vo_clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    recall = recall_score(y_test,pred)
    precision = precision_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    print("acc : {} / precision : {} / recall : {} / f1 : {}".format(acc, precision, recall, f1))

# XGBoost 실행 함수
def emsemble_xgboost(Xtrain_scaled, y_train_scaled, Xtest_scaled, y_test):

    dtrain = xgb.DMatrix(data = Xtrain_scaled, label = y_train_scaled) # 학습 데이터를 XGBoost 모델에 맞게 변환
    dtest = xgb.DMatrix(data = Xtest_scaled) # 평가 데이터를 XGBoost 모델에 맞게 변환

    # 학습을 위한 파라미터 값 설정 
    param = {'max_depth': 10, # 결정 트리의 깊이
            'eta': 0.05, # learning rate
            'objective': 'multi:softmax', # 사용할 함수
            'num_class' : 2 # 분류해야 할 클래스의 수
    } 

    model = xgb.train(params = param, dtrain = dtrain) # 학습 진행
    preds = model.predict(dtest) # 평가 데이터 예측

    print("Precision : %f" % (precision_score(y_test, preds)))
    print("Recall : %f" % (recall_score(y_test, preds)))
    print("f1 score : %f" % (f1_score(y_test, preds)))

    # feature의 중요도를 plotting하는 함수
    #plot_importance(model)


def main() :

    seed_num = 42
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    rn.seed(seed_num)

    #데이터 불러오기
    creditcard = pd.read_csv(r"creditcard.csv")
    validation = pd.read_csv(r"validation.csv",index_col=0)
    X_test = pd.read_csv(r"X_test.csv",index_col=0)
    y_test = pd.read_csv(r"y_test.csv",index_col=0)
    X_train = pd.read_csv(r"X_train.csv",index_col=0)
    y_train = pd.read_csv(r"y_train.csv",index_col=0)

    y_val= validation['class']
    X_val = validation.drop('class', axis=1)

    # DNN 모델 학습
    Epochs = 15
    BatchSize = 2048

    # 데이터 전처리
    Xtrain_scaled, ytrain_scaled, Xval_scaled, Xtest_scaled = data_preprocessing(X_train, y_train, X_val, X_test,'V15')

    # DNN 모델 생성
    model = dnn_model(Xtrain=Xtrain_scaled, optimizer=RMSprop(learning_rate=1e-3))

    # weight balancing 비율 지정
    class_weight = {0: 0.16, 1: 0.84}

    # DNN 모델 학습
    history = model.fit(Xtrain_scaled, ytrain_scaled, batch_size = BatchSize, epochs = Epochs, validation_data=(Xval_scaled,y_val), class_weight = class_weight) # weight balancing 포함하여 학습진행

    loss_plotting(history)

    # DNN 모델 평가
    results = model.evaluate(Xtest_scaled, y_test,batch_size=BatchSize)
    print("'DNN MODEL RESULT'")
    for name, value in zip(model.metrics_names, results):
        print(name, ': ', value)   

    # 모델 저장
    model.save("AI_project.h5")

    #LGB 모델 생성
    lgb_model = LGBMClassifier(random_state = seed_num, num_leaves=16, n_estimators=600, max_depth = 4, lerarning_rate = 0.1, boost_from_average = False)
    lgb_model.fit(Xtrain_scaled, ytrain_scaled)

    #LGB 모델 예측
    y_pred = lgb_model.predict(Xtest_scaled)

    # 모델 스코어 출력
    print("'LGB MODEL RESULT'")
    print("accuracy : %f" % (accuracy_score(y_test, y_pred)))
    print("precision : %f" % (precision_score(y_test, y_pred)))
    print("recall : %f" % (recall_score(y_test, y_pred)))
    print("f1 : %f" % (f1_score(y_test, y_pred)))
    print("auc : %f" % (roc_auc_score(y_test, y_pred, average='macro')))

if __name__ == "__main__" :
    main()