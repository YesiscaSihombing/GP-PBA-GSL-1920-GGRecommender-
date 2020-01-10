#!/usr/bin/env python
# coding: utf-8

# # Recommender System for Article with Levenshtein Distance

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[43]:


ds = pd.read_csv("./articles1.csv", encoding="ISO-8859-1")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 10), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['content'].values.astype('U'))

# .fit_transform(ds['content'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}


# In[49]:


ds


# In[50]:


for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]
    
print('done!')


# In[51]:


results[row['id']]


# In[52]:


def item(id):
    return ds.loc[ds['id'] == id]['title'].tolist()[0].split(' - ')[0]


# In[53]:


# Just reads the results out of the dictionary.
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")


# In[54]:


item(28336)


# In[55]:


recommend(item_id=28336, num=5)

