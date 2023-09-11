Binary Classification
In [1]:

# Step 1 : import library
import pandas as pd
     
In [2]:

# Step 2 : import data
diabetes = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')
     
In [3]:

diabetes.head()
     
Out[3]:
	pregnancies	glucose	diastolic	triceps	insulin	bmi	dpf	age	diabetes
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
In [4]:

diabetes.info()
     
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   pregnancies  768 non-null    int64  
 1   glucose      768 non-null    int64  
 2   diastolic    768 non-null    int64  
 3   triceps      768 non-null    int64  
 4   insulin      768 non-null    int64  
 5   bmi          768 non-null    float64
 6   dpf          768 non-null    float64
 7   age          768 non-null    int64  
 8   diabetes     768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
In [5]:

diabetes.describe()
     
Out[5]:
	pregnancies	glucose	diastolic	triceps	insulin	bmi	dpf	age	diabetes
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
In [6]:

# Step 3 : define target (y) and features (X)
     
In [7]:

diabetes.columns
     
Out[7]:
Index(['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age', 'diabetes'],
      dtype='object')
In [8]:

y = diabetes['diabetes']
     
In [9]:

X = diabetes.drop(['diabetes'],axis=1)
     
In [10]:

# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)
     
In [11]:

# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape
     
Out[11]:
((537, 8), (231, 8), (537,), (231,))
In [12]:

# Step 5 : select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)
     
In [13]:

# Step 6 : train or fit model
model.fit(X_train,y_train)
     
Out[13]:
LogisticRegression(max_iter=500)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
In [14]:

model.intercept_
     
Out[14]:
array([-8.13081966])
In [15]:

model.coef_
     
Out[15]:
array([[ 1.01223082e-01,  3.60559605e-02, -2.09731012e-02,
        -2.57391062e-03, -2.04496125e-04,  8.24702415e-02,
         9.51115023e-01,  2.53544397e-02]])
In [16]:

# Step 7 : predict model
y_pred = model.predict(X_test)
     
In [17]:

y_pred
     
Out[17]:
array([0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1,
       0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
       0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
       0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1,
       0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
In [18]:

# Step 8 : model accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
     
In [19]:

confusion_matrix(y_test,y_pred)
     
Out[19]:
array([[133,  12],
       [ 41,  45]])
In [20]:

accuracy_score(y_test,y_pred)
     
Out[20]:
0.7705627705627706
In [21]:

print(classification_report(y_test,y_pred))
     
              precision    recall  f1-score   support

           0       0.76      0.92      0.83       145
           1       0.79      0.52      0.63        86

    accuracy                           0.77       231
   macro avg       0.78      0.72      0.73       231
weighted avg       0.77      0.77      0.76       231


