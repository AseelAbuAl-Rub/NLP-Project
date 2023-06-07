#!/usr/bin/env python
# coding: utf-8

# In this code, we load the data from google drive and stored it in data frames for each file then concat train negative and positive in one data frame name: train data, also concat test in one data frame name: test data frame; convert the data type of labelling to boolean. We apply a cleaning process: remove numbers, special characters, and punctuation marks then tokenized text, removed stop words and apply Lemmatization.
# As a embedding representation text, we used Word2vec to build model by using training data, then tested the test data on the trained model.After that we load google word2vec pretrained model and fine-tune it in our word2vec model.

# In[1]:


import os
import pandas as pd
import nltk
import re
import numpy as np
from nltk.tokenize import word_tokenize,sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[2]:


#@title Apply (read from drive and store in df)function
train_neg = pd.read_csv('train_neg.csv')
train_pos = pd.read_csv('train_pos.csv')

test_neg = pd.read_csv('test_neg.csv')
test_pos = pd.read_csv('test_pos.csv')

test_neg


# In[17]:


#@title Concat the data
train_data = pd.concat([train_neg, train_pos], ignore_index=True)
test_data = pd.concat([test_neg, test_pos], ignore_index=True)


# In[18]:


#@title Stop words
#Stop Words
stop_words=set(stopwords.words('english'),)


# In[19]:


#@title Remove Characters And Number Function 
def remove_num_special_char(text):
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
    return cleaned_text



# In[20]:


#@title Remove Stop Words & Lemmatization Function
def remove_stopwords_lem(text):
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(token) for token in text if token not in stop_words]
    return filtered_tokens
    


# In[21]:


#@title Apply (remove character and num) function

train_data['cleaned_text'] = train_data['Tweets'].apply(remove_num_special_char)

test_data['cleaned_text'] = test_data['Tweets'].apply(remove_num_special_char)


# In[22]:


#@title Tokenize text
train_data['tokenized_text'] = train_data['cleaned_text'].apply(lambda x: word_tokenize(x))

test_data['tokenized_text'] = test_data['cleaned_text'].apply(lambda x: word_tokenize(x))


# In[23]:


#@title Convert to lower case 
train_data['tokenized_text'] = train_data['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])

test_data['tokenized_text'] = test_data['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])


# In[24]:


#@title Apply (remove stop words & stemming) function
train_data['cleaned_tokens'] = train_data['tokenized_text'].apply(remove_stopwords_lem)

test_data['cleaned_tokens'] = test_data['tokenized_text'].apply(remove_stopwords_lem)


# In[44]:


#@title Word2vec
sentences = train_data['cleaned_tokens'].values.tolist()
train_data_model = Word2Vec(sentences=sentences, vector_size=200, window=3, min_count=1, workers=4)


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Vectorize the sentences using the Word2Vec model
vectorized_sentences = []
for sentence in train_data['cleaned_tokens']:
    sentence_vectors = [train_data_model.wv[word] for word in sentence if word in train_data_model.wv]
    if sentence_vectors:
        sentence_vector = np.mean(sentence_vectors, axis=0)
        vectorized_sentences.append(sentence_vector)
    else:
        vectorized_sentences.append(np.zeros(train_data_model.vector_size))

# Convert the list of vectors to a numpy array
X_train = np.array(vectorized_sentences)

# Assign labels to the sentences based on your classification task
y_train = train_data['flag']

# Vectorize the sentences in the test data
vectorized_test_sentences = []
for sentence in test_data['cleaned_tokens']:
    sentence_vectors = [train_data_model.wv[word] for word in sentence if word in train_data_model.wv]
    if sentence_vectors:
        sentence_vector = np.mean(sentence_vectors, axis=0)
        vectorized_test_sentences.append(sentence_vector)
    else:
        vectorized_test_sentences.append(np.zeros(train_data_model.vector_size))

# Convert the list of vectors to a numpy array
X_test = np.array(vectorized_test_sentences)

# Assign labels to the sentences in the test data
y_test = test_data['flag']

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)

train_data_model.save("trained_word2vec.model")

print("Accuracy:", accuracy)



# In[27]:


from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('word2vec.bin.gz', binary=True)


# In[34]:


import numpy as np

# Takes a sentence as input and converts it into a feature vector representation using word embeddings generated by the merged model
def vectorize2(sentence):
    words_vecs = [model[word] for word in sentence if word in model]  # capture the word vectors for all words in the model
    if len(words_vecs) == 0:
        return np.zeros(300)
    words_vecs = np.array(words_vecs)  # The collected word vectors are converted into a NumPy array.
    return words_vecs.mean(axis=0)

x_train = np.array([vectorize2(sentence) for sentence in X_train])
x_test = np.array([vectorize2(sentence) for sentence in X_test])


# In[35]:


#make logistic regression based on word2vec
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train, y_train)


# In[37]:


from sklearn.metrics import accuracy_score
y_pred2 = clf.predict(X_test2)
print('Accuracy:', accuracy_score(y_test, y_pred2))


# In[ ]:




