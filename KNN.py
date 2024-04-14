import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

iris=load_iris()


column_names=iris.feature_names
print("Columns names in the iris dataset")
print(column_names)

target_names=iris.target_names
print("Target names of iris datset")
print(target_names)

iris_df=pd.DataFrame(data=iris['data'],columns=iris['feature_names'])
print(iris_df.head())
iris_df['target']=iris['target']

X=iris_df.drop('target',axis=1)
Y=iris_df['target']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

K=3
knn=KNeighborsClassifier(n_neighbors=K)

knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)

accuracy=accuracy_score(Y_test,y_pred)
print(f"Accuracy for knn (K={K}):{accuracy}")

actual_predicted=pd.DataFrame({'Actual':Y_test,'Predicted':y_pred})
print(actual_predicted)

sample=np.array([[5.1,3.5,1.4,0.2]])
predicted_class=knn.predict(sample)[0]
predicted_species=iris.target_names[predicted_class]
print(f"Predicted class fpr the sample: {predicted_species}")

warnings.filterwarnings("ignore",message="X does not have feature names")