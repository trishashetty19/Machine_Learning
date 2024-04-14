import pandas as pd
import numpy as np

data=pd.read_csv("Enjoysport.csv")
print(data)

concepts=np.array(data)[:,:-1]
target=np.array(data)[:,-1]


def learn(concepts,target):
    specific=concepts[0].copy()
    print("Initialization of specific and general:\n")
    print(specific)
    print("General hypothesis:\n")
    general=[["?" for i in range(len(specific))] for i in range(len(specific))]
    print(general)

    for i,h in enumerate(concepts):
        if target[i]=="Yes":
            for x in range(len(specific)):
                if h[x]!=specific[x]:
                    specific[x]="?"
                    general[x][x]="?"
        if target[i]=="No":
            for x in range (len(specific)):
                if h[x]!=specific[x]:
                    general[x][x]=specific[x]
                else:
                    general[x][x]="?"
        
        print("Steps for candidate eleimination algorithm",i+1)
        print(specific)
        print(general)

    indices=[i for i, val in enumerate(general)if val==["?","?","?","?","?","?"]]

    for i in indices:
        general.remove(["?","?","?","?","?","?"])
    return specific,general
    
s_final,g_final=learn(concepts,target)
print("Final specific:",s_final,sep="\n")
print("Final general:",g_final,sep="\n")