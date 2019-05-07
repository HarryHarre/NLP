
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np


# In[2]:


import matplotlib.pyplot as plt, seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


yelp = pd.read_csv('yelp.csv')


# In[5]:


yelp.head()


# In[6]:


yelp.info()


# In[7]:


yelp.describe()


# In[8]:


yelp['text length'] = yelp['text'].apply(len)


# In[9]:


yelp.head()


# In[11]:


g = sns.FacetGrid(yelp, col="stars")
g = g.map(plt.hist, "text length",bins=50)


# In[12]:


ax = sns.boxplot(x="stars", y="text length", data=yelp,palette='rainbow')


# In[13]:


sns.countplot(x='stars',data=yelp,palette='rainbow')


# In[14]:


#yelp.groupby('stars')['cool','useful','funny','text length'].mean()
stars = yelp.groupby('stars').mean()
stars


# In[15]:


stars.corr()


# In[16]:


sns.heatmap(stars.corr(),annot=True,cmap='coolwarm')


# In[17]:


yelp_class = yelp[(yelp.stars==1) | (yelp.stars==5)]


# In[18]:


#** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**
X = yelp_class['text']
y = yelp_class['stars']


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer


# In[20]:


cv = CountVectorizer()


# In[21]:


X = cv.fit_transform(X)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[24]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[25]:


nb.fit(X_train,y_train)


# In[26]:


predictions = nb.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix


# In[28]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[29]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[30]:


from sklearn.pipeline import Pipeline


# In[31]:


pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[32]:


X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


# In[33]:


# May take some time
pipeline.fit(X_train,y_train)


# In[34]:


predictions = pipeline.predict(X_test)


# In[35]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

