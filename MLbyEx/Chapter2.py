import numpy as np 
import nltk
import spacy
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

groups = fetch_20newsgroups()
#plotting the distribution of the classes
def simple_plot():
    sns.distplot(groups.target)
    plt.show()


#BoW model - initialize CountVectorizer with 500 most frequent tokens
count_vector = CountVectorizer(max_features=500)
data_count = count_vector.fit_transform(groups.data)
#print(count_vector.get_feature_names())
#raw feature extraction has a lot of redundant information and useless information

#remove numbers and combinations of letters and numbers
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True

data_cleaned = []

for doc in groups.data:
    doc_cleaned = ' '.join(word for word in doc.split()
                                if is_letter_only(word) )    
    data_cleaned.append(doc_cleaned)

#removing stop word specified according to sklearn package
count_vector_sw = CountVectorizer(stop_words="enmglish", max_features=500)
#stem and lemmatize the words
all_names = set(names.words())
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)
lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() 
                            if is_letter_only(word) and 
                            word not in all_names)
    data_cleaned.append(doc_cleaned)

data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)