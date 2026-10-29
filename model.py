import pandas as pd 
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load dataset
stud = pd.read_csv('student-data (1).csv')

# LABEL ENCODING BINARY COLUMNS
binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus','schoolsup', 'famsup', 'paid', 'activities', 'nursery','higher', 'internet', 'romantic','passed']
le = LabelEncoder()
for col in binary_cols:
    stud[col] = le.fit_transform(stud[col])

# ONEHOT ENCODING
stud=pd.get_dummies(stud,dtype='int',drop_first=True)

# X-Y SPLIT
x = stud.drop(['passed'],axis=1)
y = stud['passed']

# PRESERVING COLUMNS FOR LATER
feature_columns = x.columns.tolist()

# TRAIN- TEST SPLIT
x_train,x_test,y_train,y_test =train_test_split(
    x,y,test_size=0.2,random_state=42
)

# ML MODELLING
model = LogisticRegression()
model.fit(x_train,y_train)

# PREDICTION AND TESTING, ACCURACY
y_pred = model.predict(x_test)
print('Accuracy is : ',accuracy_score(y_test,y_pred))

# SAVING PICKLE FILES
pickle.dump(model,open('model.pkl','wb'))   

pickle.dump(feature_columns,open('feature_columns.pkl','wb'))

print('Model Training Completed')