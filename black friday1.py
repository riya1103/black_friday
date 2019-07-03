import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import sys

df=pd.read_csv("train.csv")

df=df.drop(columns=['Unnamed: 12','Unnamed: 13'])

df.count
df2=pd.read_csv('test.csv')
df['Age'].unique().tolist()
df2['Age'].unique().tolist()

df1=df.append(df2)

df['index1']=df.index.values

df['User_ID']=df['User_ID']%1000000
df.Product_ID=df.Product_ID.str.extract('(\d+)')
df = df.set_index('User_ID')
df['User_counts'] = df.index.value_counts()

df = df.set_index('Product_ID')
df['Product_count'] = df.index.value_counts()

df = df.set_index('index1')

df=df[['User_counts','Product_count','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase']]
df.info()

from sklearn.preprocessing  import LabelEncoder
labelencoder=LabelEncoder()
df['Age']=labelencoder.fit_transform(df['Age'])


df['Stay_In_Current_City_Years'].value_counts()
#replaced value taken 5
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].replace('4+',5)
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(float)


df['Product_Category_2']=df['Product_Category_2'].astype(object)
df['Product_Category_3']=df['Product_Category_3'].astype(object)
df['Product_Category_1']=df['Product_Category_1'].astype(object)
df['Marital_Status']  =  df['Marital_Status'].astype(object)
y=df['Purchase']
df=df[['User_counts','Product_count','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']]
df=pd.get_dummies(df,drop_first=True)

df.info()
df.describe()
b=df1['Product_ID'].unique().tolist()
a=df1['User_ID'].unique().tolist()

df1['index1']=df1.index.values

df1['User_ID']=df1['User_ID']%1000000
df1.Product_ID=df1.Product_ID.str.extract('(\d+)')
df1.info()


df1['City_Category'].describe()

df1['City_Category'].unique()

df1=df1.set_index('User_ID')
df1['User_counts']=df1.index.value_counts()

df1 = df1.set_index('Product_ID')
df1['Product_count']=df1.index.value_counts()

df1 = df1.set_index('index1')

df1=df1[['User_counts','Product_count','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase']]
df1.info()

from sklearn.preprocessing  import LabelEncoder
labelencoder=LabelEncoder()
df1['Age']=labelencoder.fit_transform(df1['Age'])



corr=df1.corr()
sbn.heatmap(corr,annot=True)

df1['Stay_In_Current_City_Years'].value_counts()
#replaced value taken 5
df1['Stay_In_Current_City_Years']=df1['Stay_In_Current_City_Years'].replace('4+',5)
df1['Stay_In_Current_City_Years']=df1['Stay_In_Current_City_Years'].astype(float)
X=pd.DataFrame(df1[['User_counts','Product_count','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']])

X['Age'].unique().tolist()

X['Product_Category_2']=X['Product_Category_2'].astype(object)
X['Product_Category_3']=X['Product_Category_3'].astype(object)
X['Product_Category_1']=X['Product_Category_1'].astype(object)

X['Product_Category_2'].unique().tolist()
X['Product_Category_2'].value_counts().plot.bar()

X['Product_Category_3'].value_counts().plot.bar()
X.info()
X['Marital_Status']=X['Marital_Status'].astype(object)



'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc_y = StandardScaler()
y=sc_y.fit_transform(y)
X=sc.fit_transform(X)'''

y=df['Purchase']



#calculating rmse
rss=((y-y_pred)**2).sum()
mse=np.mean((y-y_pred)**2)
rmse1=np.sqrt(np.mean((y-y_pred)**2))

df1.max().count()
df1.boxplot()
df1.columns


for column in df:
    plt.figure()
    df.boxplot([column])
    
for column in df2:
    plt.figure()
    df2.boxplot([column])

for column in df1:
    plt.figure()
    df1.boxplot([column])





X.shape
y.shape

y=df['Purchase']


#dropping marital status and product category 3
X = X[['User_counts', 'Product_count', 'Gender', 'Age', 'Occupation',
       'City_Category', 'Stay_In_Current_City_Years',
       'Product_Category_1', 'Product_Category_2']]

# Splitting the dataset into the Training set and Test set(not by the usual technique)
X_train=pd.DataFrame
X_train = X.iloc[0:550068]
X_test=X.iloc[550068:783667]

#Feature Selection using Univariate Selection

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(df,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

#Using Feature Importance

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier(n_estimators=1)
model.fit(df,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#Due to memory issues both of the above techniques did not get executed
corr= df.corr()
sbn.heatmap(corr, annot = True)

y_train=y

X_train.info()

X_train.columns

X=pd.get_dummies(X,drop_first=True)

X_train = X.iloc[0:550068]
X_test=X.iloc[550068:783667]

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBRegressor
regressor = XGBRegressor(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=6,
 gamma=0,
  reg_alpha=0.005,
 subsample=0.8,
 colsample_bytree=0.8,

 nthread=4,
 scale_pos_weight=1,
 seed=27)

regressor.fit(X_train,y_train)

model=XGBRegressor()
n_estimators = [ 100, 150, 200,500]
max_depth = [2, 4, 6, 8]
print(max_depth)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X_train,y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
# plot results
scores = numpy.array(means).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
pyplot.legend()
pyplot.xlabel('n_estimators')
pyplot.ylabel('Log Loss')
pyplot.savefig('n_estimators_vs_max_depth.png')

y_pred=regressor.predict(X_test)

from sklearn.metrics import f1_score
f1_score(y_test , y_pred)






df3=pd.DataFrame()
df4=pd.DataFrame()

df3=pd.read_csv('test.csv')
df4['User_ID']=df3['User_ID']
df4['Product_ID']=df3['Product_ID']
df4['Purchase']=y_pred


df4.to_csv('Predictions.csv')































































































