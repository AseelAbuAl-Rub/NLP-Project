#!/usr/bin/env python
# coding: utf-8

# In this code, we try fine-tune BERT model in our training data

# In[14]:


get_ipython().system('pip install transformers')


# In[15]:


pip install datasets


# In[16]:


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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[60]:


#@title Apply (read from drive and store in df)function
train_neg = pd.read_csv('train_neg.csv')
train_pos = pd.read_csv('train_pos.csv')

test_neg = pd.read_csv('test_neg.csv')
test_pos = pd.read_csv('test_pos.csv')

test_neg


# In[61]:


#@title Concat the data
train_data = pd.concat([train_neg, train_pos], ignore_index=True)
test_data = pd.concat([test_neg, test_pos], ignore_index=True)


# In[62]:


#@title Stop words
#Stop Words
stop_words=set(stopwords.words('english'),)


# In[63]:


#@title Remove Characters And Number Function 
def remove_num_special_char(text):
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", text)
    return cleaned_text



# In[64]:


#@title Remove Stop Words & Lemmatization Function
def remove_stopwords_lem(text):
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = [lemmatizer.lemmatize(token) for token in text if token not in stop_words]
    return filtered_tokens
    


# In[65]:


#@title Apply (remove character and num) function

train_data['cleaned_text'] = train_data['Tweets'].apply(remove_num_special_char)

test_data['cleaned_text'] = test_data['Tweets'].apply(remove_num_special_char)


# In[66]:


#@title Tokenize text
train_data['tokenized_text'] = train_data['cleaned_text'].apply(lambda x: word_tokenize(x))

test_data['tokenized_text'] = test_data['cleaned_text'].apply(lambda x: word_tokenize(x))


# In[67]:


#@title Convert to lower case 
train_data['tokenized_text'] = train_data['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])

test_data['tokenized_text'] = test_data['tokenized_text'].apply(lambda tokens: [token.lower() for token in tokens])


# In[68]:


#@title Apply (remove stop words & stemming) function
train_data['cleaned_tokens'] = train_data['tokenized_text'].apply(remove_stopwords_lem)

test_data['cleaned_tokens'] = test_data['tokenized_text'].apply(remove_stopwords_lem)


# In[69]:


#convert dataframe to list
x_train = train_data['cleaned_tokens'].tolist()
y_train = train_data['flag'].tolist()

x_test = test_data['cleaned_tokens'].tolist()
y_test = test_data['flag'].tolist()

#Convert list to hugging face
from datasets import Dataset
train_dataset = Dataset.from_dict({"text": x_train, "label": y_train})
test_dataset = Dataset.from_dict({"text": x_test, "label": y_test})


# In[70]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")


# In[78]:


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# In[79]:


tokenized_datasets_train = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets_test = test_dataset.map(tokenize_function, batched=True)


# In[ ]:





# In[ ]:




