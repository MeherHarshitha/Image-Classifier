#importing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

#csv file
data = pd.read_csv('https://raw.githubusercontent.com/MeherHarshitha/Image-Classifier/master/letter.csv')
data.head()

# (x,y) set
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

#training and test sets
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

# x and y sets
x_train.head()
y_train.head()

# Random Forest Classifier 
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
pred

#testing the model
s = y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
        
# Printing the count,pred and accuracy
print(count)
print(len(pred))
print("Accuracy = ",count/len(pred))
