import numpy as np
import pandas as pd 

from sklearn.naive_bayes import GaussianNB  
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('PlayTennis.csv')

label_encoder=LabelEncoder()

for col in data.columns:
    data[col]=label_encoder.fit_transform(data[col])
    
print(data)

test_data = data

X_train=data.drop('PlayTennis',axis=1)
Y_train=data['PlayTennis']

X_test = test_data.drop('PlayTennis', axis=1)
Y_test = test_data['PlayTennis']

nb_classifier=GaussianNB()
nb_classifier.fit(X_train,Y_train)

Y_pred=nb_classifier.predict(X_test)

print("Predicted Output")
for i, prediction in enumerate(Y_pred):
    print(f"Sample{i+1}:{'Yes' if prediction==1 else 'No'}")
accuracy=metrics.accuracy_score(Y_test,Y_pred)
print("Accuracy:",accuracy)
