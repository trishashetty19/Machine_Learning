import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

data=pd.read_csv("PlayTennis.csv")

features=data.drop('PlayTennis',axis=1)
target=data['PlayTennis']

encoder=LabelEncoder()
for col in features.columns:
    features[col]=encoder.fit_transform(features[col])
target=encoder.fit_transform(target)

clf=DecisionTreeClassifier(criterion="entropy")
clf.fit(features,target)

visualization=export_text(clf,feature_names=list(features),class_names=['No','Yes'])
print(visualization)
