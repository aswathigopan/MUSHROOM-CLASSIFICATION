#!/usr/bin/env python
# coding: utf-8

# # MUSHROOM CLASSIFICATION

# In[116]:


# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[117]:


# Impoer datasets
mush_df=pd.read_csv("D:/ICT TRAINING/ASSIGNMENTS/ASSIGNMENT ON MUSHROOM DATA/mushrooms.csv")


# In[118]:


# Analyse the data
mush_df.head()


# In[119]:


mush_df.shape


# In[120]:


mush_df.tail()


# In[121]:


mush_df.info()


# In[122]:


mush_df.describe()


# In[123]:


# Checking for NaN values
mush_df.isna().sum()


# In[124]:


#Analysis of target and feature variables


# In[125]:


sns.countplot(mush_df['class']);


# In[126]:


mush_df['class'].value_counts()


# In[127]:


mush_df['class'].value_counts()/len(mush_df['class'])


# In[128]:


#The data set contains almost equal count of two class labels.


# In[129]:


# Univariate Analysis


# In[130]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(mush_df['cap-shape']);
plt.subplot(1,3,2)
sns.countplot(mush_df['cap-surface']);
plt.subplot(1,3,3)
sns.countplot(mush_df['cap-color']);


# In[131]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(mush_df['bruises']);
plt.subplot(1,2,2)
sns.countplot(mush_df['odor']);


# In[132]:


plt.figure(figsize=(15,3))
plt.subplot(1,4,1)
sns.countplot(mush_df['gill-attachment']);
plt.subplot(1,4,2)
sns.countplot(mush_df['gill-spacing']);
plt.subplot(1,4,3)
sns.countplot(mush_df['gill-color']);
plt.subplot(1,4,4)
sns.countplot(mush_df['gill-size']);


# In[133]:


plt.figure(figsize=(15,8))
plt.subplot(2,3,1)
sns.countplot(mush_df['stalk-shape']);
plt.subplot(2,3,2)
sns.countplot(mush_df['stalk-root']);
plt.subplot(2,3,3)
sns.countplot(mush_df['stalk-surface-above-ring']);
plt.subplot(2,3,4)
sns.countplot(mush_df['stalk-surface-below-ring']);
plt.subplot(2,3,5)
sns.countplot(mush_df['stalk-color-above-ring']);
plt.subplot(2,3,6)
sns.countplot(mush_df['stalk-color-below-ring']);


# In[134]:


# Its clear that in the data set,the feature "stalk root" has some invalid value"?"


# In[135]:


mush_df['stalk-root'].value_counts()


# In[136]:


mush_df['stalk-root'].replace("?",mush_df['stalk-root'].mode()[0],inplace=True)


# In[137]:


mush_df['stalk-root'].value_counts()


# In[138]:


sns.countplot(mush_df['stalk-root']);


# In[139]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.countplot(mush_df['veil-type']);
plt.subplot(1,2,2)
sns.countplot(mush_df['veil-color']);


# In[140]:


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.countplot(mush_df['ring-number']);
plt.subplot(1,2,2)
sns.countplot(mush_df['ring-type']);


# In[141]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(mush_df['spore-print-color']);
plt.subplot(1,3,2)
sns.countplot(mush_df['population']);
plt.subplot(1,3,3)
sns.countplot(mush_df['habitat']);


# In[142]:


# Analysis of feature variabels with class labels


# In[143]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(mush_df['cap-shape'],hue=mush_df['class']);
plt.subplot(1,3,2)
sns.countplot(mush_df['cap-surface'],hue=mush_df['class']);
plt.subplot(1,3,3)
sns.countplot(mush_df['cap-color'],hue=mush_df['class']);


# In[144]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(mush_df['bruises'],hue=mush_df['class']);
plt.subplot(1,2,2)
sns.countplot(mush_df['odor'],hue=mush_df['class']);


# In[145]:


plt.figure(figsize=(15,3))
plt.subplot(1,4,1)
sns.countplot(mush_df['gill-attachment'],hue=mush_df['class']);
plt.subplot(1,4,2)
sns.countplot(mush_df['gill-spacing'],hue=mush_df['class']);
plt.subplot(1,4,3)
sns.countplot(mush_df['gill-color'],hue=mush_df['class']);
plt.subplot(1,4,4)
sns.countplot(mush_df['gill-size'],hue=mush_df['class']);


# In[146]:


plt.figure(figsize=(15,8))
plt.subplot(2,3,1)
sns.countplot(mush_df['stalk-shape'],hue=mush_df['class']);
plt.subplot(2,3,2)
sns.countplot(mush_df['stalk-root'],hue=mush_df['class']);
plt.subplot(2,3,3)
sns.countplot(mush_df['stalk-surface-above-ring'],hue=mush_df['class']);
plt.subplot(2,3,4)
sns.countplot(mush_df['stalk-surface-below-ring'],hue=mush_df['class']);
plt.subplot(2,3,5)
sns.countplot(mush_df['stalk-color-above-ring'],hue=mush_df['class']);
plt.subplot(2,3,6)
sns.countplot(mush_df['stalk-color-below-ring'],hue=mush_df['class']);


# In[147]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.countplot(mush_df['veil-type'],hue=mush_df['class']);
plt.subplot(1,2,2)
sns.countplot(mush_df['veil-color'],hue=mush_df['class']);


# In[148]:


plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
sns.countplot(mush_df['ring-number'],hue=mush_df['class']);
plt.subplot(1,2,2)
sns.countplot(mush_df['ring-type'],hue=mush_df['class']);


# In[149]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
sns.countplot(mush_df['spore-print-color'],hue=mush_df['class']);
plt.subplot(1,3,2)
sns.countplot(mush_df['population'],hue=mush_df['class']);
plt.subplot(1,3,3)
sns.countplot(mush_df['habitat'],hue=mush_df['class']);


# In[150]:


# We can now encode the class labels "e":0,"p":1


# In[151]:


mush_df['class']=mush_df['class'].map({'p':1,'e':0})


# In[152]:


# Before moving to one hot encoding,we can split the feature and target variables


# In[153]:


x=mush_df.drop(['class'],axis=1)
y=mush_df['class']


# In[154]:


# Perform one hot encoding to the feature selection


# In[155]:


x=pd.get_dummies(x)


# In[156]:


x.head()


# In[157]:


y.head()


# ### Here all features and target are catagorical variables and therefore we can for a chi-square test to find the effect of one variable on another

# #### Null Hypothesis:There is no relationship between the catagorical variables(features and target)
# #### Alternate Hypothesis:There is relationship between the catagorical variables(features and target)

# In[158]:


pd.crosstab(x['cap-shape_b'],y)


# In[159]:


from scipy.stats import chi2_contingency
chi2_contingency(pd.crosstab(x['cap-shape_b'],y))


# In[160]:


# By checking the p value , we can select the feature selection using chi square test


# In[161]:


# Define function to perform chi-square test and to display the independant features


# In[162]:


def check(table):
    stat,p,dof,expected=chi2_contingency(table)
    if p>0.05:
        print("The feature {}  with p value is {} ".format(i,p))        
    return expected


# In[163]:


# Create cross table and perform the chi-square test using the function
# The condition is that the feature combination  with p value greater than 0.05 are independant.


# In[164]:


print("Independent features are") 
for i in x.columns: 
    data= x[i]
    cross_table=pd.crosstab(data,y)
    check(cross_table)


# In[165]:


## So we can drop the independant column features from the dataset


# In[166]:


x.drop(['cap-shape_c','cap-shape_f','cap-surface_g','stalk-surface-above-ring_y','veil-type_p'],axis=1,inplace=True)


# In[167]:


x.shape


# In[168]:


x.head()


# In[169]:


# Split the data to test and train 


# In[170]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=32,test_size=0.3)


# In[171]:


x_train.shape


# In[172]:


y_train.shape


# In[173]:


x_test.shape


# In[174]:


y_test.shape


# In[175]:


# Since its a classification problem ,we can go for some classification algorithms


# ### LogisticRegression

# In[176]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
y_pred_log=log.predict(x_test)


# In[177]:


## Check both the test and train accuracy


# In[178]:


log.score(x_train,y_train)


# In[179]:


log.score(x_test,y_test)


# In[180]:


# Check the model accuracy


# In[181]:


accuracy_score(y_test,y_pred_log)


# In[182]:


confusion_matrix(y_test,y_pred_log)


# In[183]:


print(classification_report(y_test,y_pred_log))


# ### Decisiontree Classfier

# In[184]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)


# In[185]:


## Check both the test and train accuracy


# In[186]:


dt.score(x_train,y_train)


# In[187]:


dt.score(x_test,y_test)


# In[188]:


# Check the model accuracy


# In[189]:


accuracy_score(y_test,y_pred_dt)


# ### RandomForest Classifier

# In[190]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)


# In[191]:


# Check the train and test dataset


# In[192]:


rf.score(x_train,y_train)


# In[193]:


rf.score(x_test,y_test)


# In[194]:


# Check the model accuracy


# In[195]:


accuracy_score(y_test,y_pred_rf)


# In[196]:


pd.DataFrame(rf.feature_importances_,index=x.columns)


# In[197]:


## Naive Bayes


# In[228]:


## Gaussian NB


# In[199]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred_nb=nb.predict(x_test)


# In[200]:


# Check the train and test acuracy


# In[201]:


nb.score(x_train,y_train)


# In[202]:


nb.score(x_test,y_test)


# In[203]:


## Check the model accuracy


# In[204]:


accuracy_score(y_test,y_pred_nb)


# In[205]:


# Bernoulli NB


# In[226]:


from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB(alpha=0.02,binarize=0.0, class_prior=None, fit_prior=True)
bnb.fit(x_train,y_train)
y_pred_bnb=bnb.predict(x_test)


# In[227]:


accuracy_score(y_test,y_pred_bnb)


# In[229]:


## Multinomial NB


# In[249]:


from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB(alpha=0.001,fit_prior=True, class_prior=None)
mnb.fit(x_train,y_train)
y_pred_mnb=mnb.predict(x_test)


# In[250]:


accuracy_score(y_test,y_pred_mnb)


# ### SVM

# In[208]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(x_train,y_train)
y_pred_sv=sv.predict(x_test)


# In[209]:


accuracy_score(y_test,y_pred_sv)


# ### Bagging Classifier

# In[210]:


from sklearn.ensemble import BaggingClassifier
bc=BaggingClassifier()
bc.fit(x_train,y_train)
y_pred_bc=bc.predict(x_test)


# In[211]:


accuracy_score(y_test,y_pred_bc)


# ### Adaboost Classifier

# In[212]:


from sklearn.ensemble import AdaBoostClassifier
ad=AdaBoostClassifier()
ad.fit(x_train,y_train)
y_pred_ad=ad.predict(x_test)


# In[213]:


accuracy_score(y_test,y_pred_ad)


# ### GradientBoost Classifier

# In[214]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
gb.fit(x_train,y_train)
y_pred_gb=gb.predict(x_test)


# In[215]:


accuracy_score(y_test,y_pred_gb)


# ### Extreme GradientBoost Classifier

# In[216]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
y_pred_xgb=xgb.predict(x_test)


# In[217]:


accuracy_score(y_test,y_pred_xgb)


# ### KNN

# In[218]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)


# In[219]:


accuracy_score(y_test,y_pred_knn)

