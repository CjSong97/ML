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
from sklearn.manifold import TSNE

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
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)
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

# visualizing t-SNE
def vis_tsne():
    categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
    groups_3 = fetch_20newsgroups(categories=categories_3)
    #cleaning the data
    data_cleaned_3 = []
    for doc in groups_3.data:
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() 
                            if is_letter_only(word) and 
                            word not in all_names)  
        data_cleaned_3.append(doc_cleaned)
    count_vector_3_sw = CountVectorizer(stop_words="english", max_features=500)
    
    data_cleaned_count_3 = count_vector_3_sw.fit_transform(data_cleaned_3)
    tsne_model = TSNE(n_components=2, perplexity=40,
                      random_state=42, learning_rate=500)
    
    data_tsne = tsne_model.fit_transform(data_cleaned_count_3.toarray())
    plt.scatter(data_tsne[:, 0], data_tsne[:,1], c=groups_3.target)
    plt.show()
    
    #maintaining similarity with count vectorization
    categories_5 = ['comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 
                    'comp.windows.x']
    groups_5 = fetch_20newsgroups(categories=categories_5)
    
vis_tsne()