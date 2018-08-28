##8/28/18 
import keras 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import SGD 
from keras.constraints import maxnorm 
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import numpy as np 
import pandas as pd 

food=pd.read_csv("compounds.csv",encoding="latin-1")
food=pd.DataFrame(food)
food['klass'].value_counts()

#assign klass to number
food.klass=pd.Categorical(food.klass)
food['new_klass']=food.klass.cat.codes
food['new_klass'].value_counts()
food['protein_weight'].mean()

#drop rows without a klass identifier
food1=food[food['new_klass']>0]

#subset data 
variables=food1[["moldb_alogps_logp","moldb_logp","moldb_alogps_logs","moldb_pka","moldb_average_mass","moldb_mono_mass","new_klass"]]
variables.info()
variables=variables.dropna() 
variables1=variables[["moldb_alogps_logp","moldb_logp","moldb_alogps_logs","moldb_pka","moldb_average_mass","moldb_mono_mass","new_klass"]].values
X=variables1[:,0:6]
y=variables1[:,6]

#train and test sets 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

X_train.shape[1] #6 input variables

#multi-perceptron neural network 
model=Sequential()
model.add(Dense(4,input_dim=6,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4,activation='relu'))
model.add(Dense(219,activation='softmax'))
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
optimizer=sgd,metrics=['accuracy'])

model.fit(X_train,y_train,epochs=5,batch_size=40) 
model_score=model.evaluate(X_test,y_test,batch_size=40)
model_score[0]






