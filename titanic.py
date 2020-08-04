
import numpy as np
import pandas as pd


d= pd.read_csv('t.csv')
df=pd.read_csv('test.csv')
d.drop(['Name','Ticket'],axis=1,inplace=True)
sex=pd.get_dummies(d['Sex'],drop_first=True)
embark=pd.get_dummies(d['Embarked'],drop_first=True)
d=pd.concat([d,sex,embark],axis=1)
d.drop(['Sex','Embarked','Cabin'],axis=1,inplace=True)
   

   
   
#X=d.iloc[:,2:11].values
X=d.iloc[:,[0,2,5,6,7,8,9]].values
Y=d.iloc[:,1:2].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean',verbose=0)
imputer.fit(X[:,2:11],y=None)
X[:,1:3]=imputer.fit_transform(X[:,1:3])  

  
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(X,Y, test_size=0.4,random_state=101)

from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
pre=log.predict(x_test)
from sklearn.naive_bayes import GaussianNB
classifier= GaussianNB()
pre_a=classifier.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,pre))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pre)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
rfc_p=rfc.predict(x_test)
confusion_matrix(y_test,rfc_p) 


submission=pd.DataFrame({"PassengerId":df['PassengerId'], "Survived":pre[0]})
submission.to_csv("Kaggle.csv",index=False)