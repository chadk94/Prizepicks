import pandas as pd

def dataload(file):
    with open(file) as f:
        print(f)
    data=pd.read_csv(file,encoding='cp1252')
    data=data.dropna()
    return data

#todo isolate player for regression
#categorize variables
#categorize opponents/home/name?
#filter to just mls
#setup neural net to predict shots
#create interface to compare vs.prizepicks