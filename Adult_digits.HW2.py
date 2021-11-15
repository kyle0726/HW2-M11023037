#!/usr/bin/env python
# coding: utf-8

# # KNN

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[20]:


digits=datasets.load_digits()
X=digits.data
y=digits.target


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=1)


# In[22]:


clf=KNeighborsClassifier(n_neighbors=3,p=2,weights='distance',algorithm='brute')
clf.fit(X_train,y_train)


# In[23]:


clf.predict(X_test)


# In[24]:


clf.score(X_test,y_test)


# In[25]:


accuracy = []
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    
k_range = range(1,100)
plt.plot(k_range, accuracy)
plt.show()


# # SVR

# In[58]:


from sklearn import datasets
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# In[59]:


digits = datasets.load_digits()
X=digits.data
y=digits.target


# In[60]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)


# In[61]:


clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf.fit(X, y)


# In[62]:


clf.fit(digits.data, y)
predict=clf.predict(X)
clf.score(X, y)


# In[63]:


plt.scatter(predict,y,s=2)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Measured')


# # Random Forest

# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[50]:


digits = datasets.load_digits()
X=digits.data
y=digits.target


# In[51]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)


# In[52]:


rfc=RandomForestClassifier(n_estimators=5)
rfc.fit(X_train,y_train)


# In[53]:


y_predict=rfc.predict(X_test)
y_predict


# In[54]:


rfc.score(X_test,y_test)


# In[55]:


imp=rfc.feature_importances_


# In[56]:


names=iris.feature_names


# In[57]:


zip(imp,names)
imp, names= zip(*sorted(zip(imp,names)))
plt.barh(range(len(names)),imp,align='center')
plt.yticks(range(len(names)),names)
plt.xlabel('Importance of Features')
plt.ylabel('Features')
plt.title('Importance of Each Feature')
plt.show()


# # xgboost

# In[68]:


from sklearn.ensemble import RandomForestClassifier


# In[69]:


digits = datasets.load_digits()
X=digits.data
y=digits.target


# In[70]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8)


# In[71]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[72]:


from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)


# In[73]:


print('The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test))


# In[ ]:




