import boto3
import time
from io import StringIO
import pandas as pd
import numpy as np
import time
import string 
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import python files
import preprocess_data
import store_and_load_functions
import vectorize_templates


def load_proper_templates(year):
    vect = store_and_load_functions.load_pickle(path='nDR-page-prediction/vectorized_templates/' + str(year) + '/', file_name='vect')
    vect_templates = store_and_load_functions.load_pickle(path='nDR-page-prediction/vectorized_templates/' + str(year) + '/', file_name='vect_templates')
    templates_columns = store_and_load_functions.load_pickle(path='nDR-page-prediction/vectorized_templates/' + str(year) + '/', file_name='templates_columns')
    
    return vect, vect_templates, templates_columns


def vectorize_test_data(doc, vect):
    vect_doc = vect.transform(doc['text'])
    
    return vect_doc
    
    
def calculate_similarity(vect_doc, vect_templates, templates_columns):
    # calculate similarity
    similarities = cosine_similarity(vect_doc, vect_templates)
    similarities = pd.DataFrame(similarities).round(3)
    similarities.columns = templates_columns
    
    # get top two template similarity names per document
    arr = np.argsort(-similarities.values, axis=1)
    top_templates = pd.DataFrame(similarities.columns[arr], index=similarities.index).iloc[:, 0:2]
    
    # get top two template similarity values per document
    top_values = (-similarities.iloc[:, :-1]).apply(np.sort, axis=1).apply(lambda x: x[:2]).apply(pd.Series)*-1    
    
    # merge top templates and values
    similarities = pd.merge(top_templates, top_values, left_index=True, right_index=True)
    similarities.columns = ['y_pred_raw', 'y_pred_raw_2nd', 'y_pred_similarity', 'y_pred_similarity_2nd']
    
    return similarities

def merge_similarities_and_doc(similarities, doc):
    # merge back to fields we care about
    similarities = pd.merge(doc.reset_index()[['index', 'file_name', 'target', 'year', 'page', 'text']],
                            similarities, how='inner', left_index=True, right_index=True)

    return similarities
    

def main(docs, templates):
    start = time.time()
    year = doc['year'].unique()[0]
    vect, vect_templates, templates_columns = load_proper_templates(year)
    vect_doc = vectorize_test_data(doc, vect)
    similarities = calculate_similarity(vect_doc, vect_templates, templates_columns)
    similarities = merge_similarities_and_doc(similarities, doc)

    print("cosine_similarity complete in {}s".format(round(time.time() - start, 4)))
    return similarities

if __name__ == '__main__':
    vectorize_templates_bool = True
    
    docs, templates = preprocess_data.main()
    
    if vectorize_templates_bool:
        vectorize_templates.main(templates)
    
    similarities_all = pd.DataFrame()
    
    for file_name, doc in docs.groupby(['file_name']):
        similarities = main(docs, templates)
        similarities_all = similarities_all.append(similarities)

    similarities_all.to_csv('nDR-page-prediction/results/similarities.csv')