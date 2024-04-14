import pandas as pd
import numpy as np

data=pd.read_csv("Enjoysport.csv")

concepts=np.array(data)[:,:-1]
target=np.array(data)[:,-1]

def train(concepts,target):
    for i,val in enumerate(target):
        if val=="Yes":
            specific=concepts[i].copy()
            break
    for i,val in enumerate(concepts):
        if target[i]=="Yes":
            for x in range (len(specific)):
                if val[x]!=specific[x]:
                    specific[x]="?"
                else:
                    pass
    return specific
print(train(concepts,target))

