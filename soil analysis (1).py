#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install ipywidgets


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import metrics 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
PATH = './'
FILENAME = 'Crop_recommendation (1).csv'
data = pd.read_csv(PATH+FILENAME)
data.head()
data.info()
 #for which crops do we have data
crop_names = data['label'].unique()
print(crop_names)

# how many types of crops are there in the dataset
print(data['label'].unique().shape)
# how many data points do we have per crop
data['label'].value_counts()
# do we have missing data
data.isnull().sum()
# nope, all good!
# let's introduce more meaningful labels 
data.rename(columns={'N':'nitrogen','P':'phosphorus','K':'potassium','label':'crop'}, inplace=True)
data.head()
 #create variables that define what our dependent and independent variables are
features = ['nitrogen','phosphorus','potassium','temperature','humidity','ph','rainfall']
target = ['crop']
# let's split the data up into features and labels
X = data[features]
y = data[target]
# test size defaults to 25% of whole dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.33)
for ii, col in enumerate(features):
  print('{} (min,max): \t \t {:.2f} {:.2f}'.format(col,data[col].min(),data[col].max()))
# scale inputs. It's important that we apply the scaling after splitting data into training and test
# Otherwise, we would introduce a bias in the training, as the scaling would depend on the test data which 
# in practice is not available during training
mmscaler = MinMaxScaler() 
X_train = mmscaler.fit_transform(X_train)
X_test = mmscaler.transform(X_test)
# convert labels to numerical values 
y_train = LabelEncoder().fit_transform(np.asarray(y_train).ravel())
y_test = LabelEncoder().fit_transform(np.asarray(y_test).ravel())
for ii, col in enumerate(features):
  print('{} (min,max): \t \t {:.2f} {:.2f}'.format(col,X_train[:,ii].min(),X_train[:,ii].max()))
 #define our model. This is a one-liner, thanks to the powerful machine learning library scikit-learn.
model = RandomForestClassifier()
# fit the model to the training data 
model.fit(X_train,y_train)
# get predictions on the test data 
y_pred=model.predict(X_test)
# print training and test accuracy
print('Training Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_train,model.predict(X_train))*100,metrics.accuracy_score(y_test,model.predict(X_test))*100))
from sklearn.metrics import confusion_matrix
y_pred=model.predict(X_test)
metrics.accuracy_score(y_test,y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_pred,y_test),square=True,cmap='Blues_r',annot=True,fmt=".0f",linewidths=.5)
ax = plt.gca()
_ = ax.set_xticklabels(crop_names,rotation='vertical')
_ = ax.set_yticklabels(crop_names,rotation='horizontal')
plt.tight_layout()
print(metrics.classification_report(y_pred,y_test))
from sklearn import tree

# fit a smaller forest with a maximum depth of 3 (this is how many consecutive 
# decision the algorithm can make). As a consequence, the accuracy will be lower
# but it'll be easier to visualise it
small_rf = RandomForestClassifier(max_depth=3)
# fit the forest to the training data 
small_rf.fit(X_train,y_train)
# get predictions on the test data 
y_pred=small_rf.predict(X_test)
# print training and test accuracy
print('Training Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_train,small_rf.predict(X_train))*100,metrics.accuracy_score(y_test,small_rf.predict(X_test))*100))


# obtain list of decision trees
trees = small_rf.estimators_
# how many are there
print('there are {n} trees in the forest'.format(n=len(trees)))

# visualise the first tree 
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20),dpi=300)
tree.plot_tree(trees[0],
               feature_names = features, 
               class_names=crop_names,
               filled = True);
import ipywidgets as widgets
def get_predictions(x1,x2,x3,x4,x5,x6,x7):
    feature = mmscaler.transform(np.asarray([x1,x2,x3,x4,x5,x6,x7]).reshape((1,-1)))
    croptoplant = crop_names[model.predict(feature).item()]
    print('{} should grow very well under these conditions'.format(croptoplant.upper()))
N = widgets.FloatSlider(min=0.0, max=140.0, value=25.0, step=2.5, description="Nitrogen")
P = widgets.FloatSlider(min=5.0, max=145.0, value=25.0, step=2.5, description="Phosphorus")
K = widgets.FloatSlider(min=5.0, max=205.0, value=25.0, step=2.5, description="Potassium")
temp = widgets.FloatSlider(min=10.0, max=44.0, value=25.0, step=2.5, description="Temperature")
hum = widgets.FloatSlider(min=15.0, max=99.0, value=25.0, step=2.5, description="humidity")
ph = widgets.FloatSlider(min=3.5, max=9.9, value=5.0, step=.5, description="pH")
rain = widgets.FloatSlider(min=20.0, max=298.0, value=25.0, step=2.5, description="Rainfall (mm)")
    
im = widgets.interact_manual(get_predictions,x1=N,x2=P,x3=K,x4=temp,x5=hum,x6=ph,x7=rain)
_ = im.widget.children[-2].description = 'get prediction'
_ = im.widget.children[-2].style.button_color='lightgreen'

display(im)

selector = SelectKBest(score_func=f_classif,k='all')
X_train_kbest = selector.fit_transform(X_train,np.asarray(y_train).ravel())
scores = selector.scores_


X_test_kbest = selector.transform(X_test)
mask = selector.get_support() #list of booleans
new_features = [] 
scores = scores[mask==True]
for bool, feature in zip(mask, features):
    if bool:
        new_features.append(feature)
        
_ = sns.barplot(x=new_features,y=scores,log=True)
plt.ylabel('Univariate score')
plt.xlabel('predictor' )
_= plt.xticks(ticks=np.arange(X_train_kbest.shape[-1]),labels=new_features,rotation='vertical')
selector = SelectKBest(score_func=f_classif,k=5)
X_train_kbest = selector.fit_transform(X_train,np.asarray(y_train).ravel())
scores = selector.scores_


X_test_kbest = selector.transform(X_test)
mask = selector.get_support() #list of booleans
new_features = [] 
scores = scores[mask==True]
for bool, feature in zip(mask, features):
    if bool:
        new_features.append(feature)

# check if it dropped temperature and ph        
print(new_features)

# run training and test on reduced dataset 
model_reduced = RandomForestClassifier()
model_reduced.fit(X_train[:,mask],y_train)

# print training and test accuracy
print('all features: Training Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_train,model.predict(X_train))*100,metrics.accuracy_score(y_test,model.predict(X_test))*100))
print('fewer features: Training Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(metrics.accuracy_score(y_train,model_reduced.predict(X_train[:,mask]))*100,metrics.accuracy_score(y_test,model_reduced.predict(X_test[:,mask]))*100))
models = []
models.append(('LogisticRegression',LogisticRegression(max_iter=5000)))
models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
models.append(('XGBClassifier',XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')))
models.append(('GradientBoostingClassifier',GradientBoostingClassifier()))
models.append(('RandomForestClassifier',RandomForestClassifier()))
models.append(('KNeighborsClassifier',KNeighborsClassifier()))
models.append(('GaussianNB',GaussianNB()))
models.append(('SVM',SVC()))

# same as above, but in cross-validation
nfolds = 5
print('{} fold cv'.format(nfolds))
X_cv = np.asarray(X)
y_cv = LabelEncoder().fit_transform(np.asarray(y).ravel())

for name,model in models:
    # apply transformation to each individual fold
    pipeline = Pipeline([('transformer', MinMaxScaler()), ('estimator', model)])    
    scores = cross_val_score(pipeline, X_cv,y_cv , cv=nfolds)
    print(name, np.round(scores.mean(),3))


# In[ ]:




