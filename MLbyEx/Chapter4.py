import glob
import os
import numpy as np
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

#implementing naive bayes from scratch

#downloaded Enron email dataset
emails, labels = [], []

#load spam
file_path = './MLbyEx/enron1/spam'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)
        
file_path = './MLbyEx/enron1/ham'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)
        
#cleaning data
def is_letter_only(word):
    return word.isalpha()
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    docs_cleaned = []
    for doc in docs:
        doc = doc.lower()
        doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                                for word in doc.split() if is_letter_only(word)
                                and word not in all_names)
        docs_cleaned.append(doc_cleaned)
    return docs_cleaned
emails_cleaned = clean_text(emails)

cv = CountVectorizer(stop_words="english", max_features=1000, max_df=0.5, min_df=2)
docs_cv = cv.fit_transform(emails_cleaned)

#start with prior, group data by label and record index of samples
def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index
label_index = get_label_index(labels)

def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior
prior = get_prior(label_index)

def get_likelihood(term_matrix, label_index, smoothing=0):
    """
    Compute likelihood based on training samples
    @param term_matrix: sparse matrix of the term frequency features
    @param label_index: grouped sample indices by class
    @param smoothing: integer, additive Laplace smoothing parameter
    @return: dictionary, with class as key, corresponding conditional probability
             P(feature | class) vector as value
    """
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood
smoothing = 1
likelihood = get_likelihood(docs_cv, label_index, smoothing)
print(len(likelihood[0]))

def get_posterior(term_matrix, prior, likelihood):
    """
    Compute posterior of testing samples, based on prior and likelihood
    @param term_matrix: sparse matrix of the term frequence features
    @param prior: dictionary, with class label as key, corresponding prior as value
    @param likelihood: dictionary, class label as key, corresponding conditional
                       probability vector as value
    @return: dictionary, with class label as key, corresponding posterior as value
    """
    num_docs = term_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label) for key, 
                     prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
            min_log_posterior = min(posterior.values())
            for label in posterior:
                try:
                    posterior[label] = np.exp(posterior[label]-min_log_posterior)
                except:
                    posterior[label] = flow('inf')
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors
        