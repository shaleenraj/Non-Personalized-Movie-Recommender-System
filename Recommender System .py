#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import os
cwd = os.getcwd()
print(cwd)


# In[79]:


#manipulating our data 
df1=pd.read_csv('tmdb_5000_credits.csv')
df2=pd.read_csv('tmdb_5000_movies.csv')
df1.columns=['id','title','cast','crew']
df2=df2.merge(df1,on='id')
df2.head(5)
df2.shape


# In[67]:


#demographic filtering 
#movies that have higher popularity will have higher probability of being liked by the user 

C=df2['vote_average'].mean()
C
#Use cutoff to include quantile 
m=df2['vote_count'].quantile(0.9)
m


# In[53]:



q_movies=df2.copy().loc[df2['vote_count']>=m]
q_movies.shape


# In[54]:


def weighted_rating(x,m=m,C=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m)*R)+(m/(m+v)*C)


# In[78]:


q_movies['score']=q_movies.apply(weighted_rating,axis=1)
q_movies.head(5)


# In[76]:


#Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending=False)

q_movies[['original_title', 'vote_count', 'vote_average', 'score']].head(10)


# In[86]:


pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['original_title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# In[102]:


from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])
print(tfidf_matrix)


# In[96]:


from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['original_title']).drop_duplicates()
print(indices.head(5))


# In[100]:


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['original_title'].iloc[movie_indices]


# In[101]:


get_recommendations('The Dark Knight Rises')


# In[135]:


from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
    
df2.head(3)


# In[136]:


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[137]:


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[138]:


df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)


# In[139]:


df2[['original_title', 'cast', 'director', 'keywords', 'genres']].head(3)


# In[140]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# In[141]:


features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)


# In[142]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)


# In[143]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[144]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[147]:


df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['original_title'])


# In[148]:


get_recommendations('The Dark Knight Rises', cosine_sim2)


# In[ ]:




