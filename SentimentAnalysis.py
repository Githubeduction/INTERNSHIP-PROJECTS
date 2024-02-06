#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:01:25 2024

@author: aggie
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
import numpy as np


import streamlit as st 
from textblob import TextBlob


df = pd.read_csv('/Users/aggie/Downloads/Amazon Product Review (4).csv') 
df.head()
df.tail()

df.head()
df.describe()
df.info()

print(df.sentiment.value_counts())

df.head()
print(df.sentiment.value_counts())
      
 # Sample text data and labels
texts = ["Text 1", "Text 2", ...]
labels = [0, 1, ...]  # 0 for one class, 1 for another class     
# Find the number of positive and negative reviews
print('Number of positive and negative reviews: ', df.sentiment.value_counts())

# Find the proportion of positive and negative reviews
print('Proportion of positive and negative reviews: ', df.sentiment.value_counts() / len(df))

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(df['review_body'])

print(type(text_counts))
print(text_counts.shape)
#print(text_counts.toarray())
print(text_counts.toarray())
#print(cv.vocabulary_)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = None )
text_counts= cv.fit_transform(df['review_body'])

print(type(text_counts))
print(text_counts.shape)
#print(text_counts.toarray())
print(text_counts.toarray())
#print(cv.vocabulary_)

# Model  building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text_counts, df['sentiment'], test_size=0.3, random_state=1)

# Model Evaluation
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
classifier = MultinomialNB().fit(X_train, y_train)
predicted= classifier.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


from sklearn.metrics import accuracy_score, classification_report

# Predict sentiment on the testing data
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report with precision, recall, F1-score, etc.
report = classification_report(y_test, y_pred)

# Output the results
print("Accuracy: ", accuracy)
print("Classification Report:\n", report)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize= (10, 6))
sns.scatterplot(x= y_test, y= predicted, color= 'blue')
plt.xlabel('Review sentiment')
plt.ylabel('Number of Review')
plt.show()
sns.heatmap(df.corr() , annot =  True,  cmap = 'coolwarm')

import matplotlib.pyplot as plt

# Count the frequency of each sentiment label
sentiment_counts = pd.Series(y_test).value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Sentiment Distribution")
plt.show()


# Create a Bar chart
import matplotlib.pyplot as plt
sentiment_count=df.groupby('sentiment').count()
plt.bar(sentiment_count.index.values, sentiment_count['star_rating'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


# Streamlit App
import streamlit as st
  
from streamlit_extras.let_it_rain import rain 

st.set_page_config(
    page_title = "Sentiment Analysis App",
    layout = "centered",
    initial_sidebar_state = "auto"
    )

st.title("AGGIE'S simple Sentiment Analysis App using Streamlit")

col1, col2 = st.columns([1, 1])
with col1: 
    st.image("/Users/aggie/Downloads/istockphoto-1371114009-1024x1024.jpg") 
with col2: 
    st.write("""Your Sentiment Analysis App""")
    
with st.echo():
    df = pd.read_csv('/Users/aggie/Downloads/Amazon Product Review (4).csv')
    st.dataframe(df)
df = pd.DataFrame(df)



  

import streamlit as st 
from textblob import TextBlob





import streamlit as st 
from textblob import TextBlob 
from streamlit_extras.let_it_rain import rain

st.title("A Simple Sentiment Analysis App.") 
message = st.text_area("Please Enter your text") 
blob = TextBlob(message) 
result = blob.sentiment 
polarity = result.polarity 
ubjectivity = result.subjectivity 
polarity = result.polarity


if st.button("Analyze the Sentiment"): 
        blob = TextBlob(message) 
        result = blob.sentiment 
        polarity = result.polarity 
        subjectivity = result.subjectivity
if polarity < 0: 
    	st.warning("The entered text has negative sentiments associated with it"+str(polarity)) 
    	rain( 
    	emoji="????", 
    	font_size=20, # the size of emoji 
    	falling_speed=3, # speed of raining 
    	animation_length="infinite", # for how much time the animation will happen 
    ) 
if polarity >= 0: 
             st.success("The entered text has positive sentiments associated with it."+str(polarity)) 
             rain( 
            	emoji="????", 
            	font_size=20, # the size of emoji 
            	falling_speed=3, # speed of raining 
            	animation_length="infinite", # for how much time the animation will happen 
            	) 
             st.success(result) 
        


 



 



 
      
        
        
        
        
        
        
        
        
        
