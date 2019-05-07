
# coding: utf-8

# In[1]:


import nltk


# In[2]:


#nltk.download_shell()


# In[3]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[4]:


print(len(messages))


# In[5]:


messages[50]


# In[6]:


for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[7]:


import pandas as pd


# In[8]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','message'])


# In[9]:


messages.head()


# In[10]:


messages.describe()


# In[11]:


messages.groupby('label').describe()


# In[12]:


messages['length'] = messages['message'].apply(len)


# In[13]:


messages.head()


# In[14]:


import matplotlib.pyplot as plt, seaborn as sns


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


messages['length'].plot.hist(bins=150)


# In[32]:


messages['length'].describe()


# In[33]:


messages[messages['length'] == 910]['message'].iloc[0]


# In[34]:


messages.hist(column='length',by='label',bins=60,figsize=(12,4))


# In[35]:


import string


# In[36]:


mess = 'Sample message! Notice: it has punctuation.'


# In[37]:


string.punctuation


# In[38]:


nopunc = [c for c in mess if c not in string.punctuation]


# In[39]:


nopunc


# In[40]:


from nltk.corpus import stopwords


# In[41]:


nopunc = ''.join(nopunc)


# In[42]:


nopunc


# In[43]:


nopunc.split()


# In[44]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[45]:


clean_mess


# In[46]:


def text_process(mess):
    """
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    """
    
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[47]:


messages.head()


# In[49]:


messages['message'].head(5).apply(text_process)


# In[50]:


from sklearn.feature_extraction.text import CountVectorizer


# In[51]:


# bow = bag of words
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[52]:


print(len(bow_transformer.vocabulary_))


# In[53]:


mess4 = messages['message'][3]


# In[54]:


print(mess4)


# In[55]:


bow4 = bow_transformer.transform([mess4])


# In[56]:


print(bow4)


# In[58]:


print(bow4.shape)


# In[59]:


bow_transformer.get_feature_names()[4068]


# In[60]:


bow_transformer.get_feature_names()[9554]


# In[61]:


messages_bow = bow_transformer.transform(messages['message'])


# In[62]:


print('Shape of Sparse Matrix: ', messages_bow.shape)


# In[63]:


messages_bow.nnz


# In[64]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[65]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))


# In[66]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[67]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[68]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[69]:


print(tfidf4)


# In[70]:


tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[72]:


messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[73]:


from sklearn.naive_bayes import MultinomialNB


# In[74]:


spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[76]:


spam_detect_model.predict(tfidf4)[0]


# In[77]:


messages['label'][3]


# In[78]:


all_pred = spam_detect_model.predict(messages_tfidf)


# In[82]:


all_pred


# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)


# In[87]:


from sklearn.pipeline import Pipeline


# In[88]:


pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier',MultinomialNB())
])


# In[89]:


pipeline.fit(msg_train,label_train)


# In[90]:


predictions = pipeline.predict(msg_test)


# In[92]:


from sklearn.metrics import classification_report


# In[93]:


print(classification_report(label_test,predictions))

