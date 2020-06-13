#k-means from scratch using iris dataset
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import datasets
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

def viz():
    y_0 = np.where(y==0)
    plt.scatter(X[y_0, 0], X[y_0, 1])
    y_1 = np.where(y==1)
    plt.scatter(X[y_1,0], X[y_1, 1])
    y_2 = np.where(y==2)
    plt.scatter(X[y_2, 0], X[y_2, 1])
    plt.show()

viz()
#specify k and initialize centroids
k = 3
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]
    
def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster

def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_1 = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis = 0)

def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*',
                s = 200 , c='#050505')
    plt.show()

#specify tolerance and maximum iterations for algorithm
tol = 0.0001
max_iter = 100
iter = 0
centroids_diff = 100000
clusters = np.zeros(len(X))

def result_from_scratch():
    while iter < max_iter and centroids_diff > tol:
        for i in range(len(X)):
            clusters[i] = assign_cluster(X[i], centroids)
            centroids_prev = deepcopy(centroids)
            iter += 1
            centroids_diff = np.linalg.norm(centroids - centroids_prev)
            print('Iteration: ' , str(iter))
            print('Centroids: \n', centroids)
            print('Centroids move: {:5.4f}'.format(centroids_diff))
            visualize_centroids(X, centroids)
        
    #plotting the clusters
    for i in range(k):
        cluster_i = np.where(clusters == i)
        plt.scatter(X[cluster_i, 0], X[cluster_i, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    #plt.show()

#k-means with scikit learn
kmeans_sk = KMeans(n_clusters=3, random_state=42)
kmeans_sk.fit(X)

#obtain clustering results, clusters for data samples and centroids of individual clusters
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_

#plot clusters along with centroids
def clusters_3():
    for i in range(k):
        cluster_i = np.where(clusters_sk == i)
        plt.scatter(X[cluster_i, 0], X[cluster_i, 1])
    plt.scatter(centroids_sk[:,0], centroids_sk[:,1],
                marker='*', s=200, c='#050505')
    plt.show()
    
#Apply Elbow method to iris dataset 
iris = datasets.load_iris()
X = iris.data
y = iris.target
k_list = list(range(1, 7))
sse_list = [0] * len(k_list)

#find the best SSE before slow down in drop-off
def sse_drop():
    for k_ind, k in enumerate(k_list):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        clusters = kmeans.labels_
        centroids = kmeans.cluster_centers_
        sse = 0
        for i in range(k):
            cluster_i = np.where(clusters == i)
            sse += np.linalg.norm(X[cluster_i] - centroids[i])
        print('k={}, SSE={}'.format(k, sse))
        sse_list[k_ind] = sse
    #plot sse vs k 
    plt.plot(k_list, sse_list)
    plt.show()
    
#cluster newsgroup using kmeans
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space'
]

groups = fetch_20newsgroups(subset='all', categories = categories)
labels = groups.target
label_names = groups.target_names
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
        return True
 
 all_names = set(names.words())
 lemmatizer = WordNetLemmatizer()
 data_cleaned = []
 for doc in groups.data:
     doc = doc.lower()
     doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for
                            word in doc.split() if is_letter_only(word)
                            and word not in all_names)
     data_cleaned.append(doc_cleaned) 
    
count_vector = CountVectorizer(stop_words='english',
                               max_features=None, max_df=0.5, min_df=2)
data = count_vector.fit_transform(data_cleaned)

#now try to cluster into four groups
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)

#using tf-idf instead for more representative approach
tfidf_vector = TfidfVectorizer(stop_words='english', max_features=None,
                               max_df=0.5, min_df=2)

data = tfidf_vector.fit_transform(data_cleaned)
kmeans.fit(data)
clusters = kmeans.labels_

#topic modelling using NMF
t = 20
nmf = NMF(n_components=t, random_state=42)
data = count_vector.fit_transform(data_cleaned)
nmf.fit(data)

#topic modelling using LDA
