import boto3
import time
import os
import sys
import inspect
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
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='nDR-page-prediction/results/log_cosine_similarity.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s:%(message)s')

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# local app imports
import app.preprocess_data as preprocess_data
import app.common as common
import app.vectorize_templates as vectorize_templates
import app.postprocess_predictions as postprocess_predictions
import app.postprocess_alerts as postprocess_alerts
import app.postprocess_performance_analysis as postprocess_performance_analysis


def load_proper_templates(year):
    
    vect = common.load_pickle(path='nDR-page-prediction/vectorized_templates/' + str(year) + '/', file_name='vect')
    vect_templates = common.load_pickle(path='nDR-page-prediction/vectorized_templates/' + str(year) + '/', file_name='vect_templates')
    templates_columns = common.load_pickle(path='nDR-page-prediction/vectorized_templates/' + str(year) + '/', file_name='templates_columns')
    
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
    similarities = pd.merge(doc.reset_index(drop=True)[['file_name', 'target', 'year', 'page', 'text']],
                            similarities, how='inner', left_index=True, right_index=True)

    return similarities
    

def main(docs, templates):
    
    start = time.time()
    
    similarities_all = pd.DataFrame()
    i = 0
    
    for file_name, doc in docs.groupby(['file_name']):
        year = doc['year'].unique()[0]
        vect, vect_templates, templates_columns = load_proper_templates(year)
        vect_doc = vectorize_test_data(doc, vect)
        similarities = calculate_similarity(vect_doc, vect_templates, templates_columns)
        similarities = merge_similarities_and_doc(similarities, doc)
        similarities_all = similarities_all.append(similarities)
        i+=1
    
    similarities_all = similarities_all.reset_index(drop=True)

    logging.info("{0}s, {1}s avg per doc, {2}s per page".format(
        round(time.time() - start, 4), round((time.time() - start)/i, 4), round((time.time() - start)/len(similarities_all), 4)))
    
    return similarities_all


if __name__ == '__main__':
    
    vectorize_templates_bool = False
    development_bool = True
    
    docs, templates = preprocess_data.main()
    
    if vectorize_templates_bool:
        vectorize_templates.main(templates)
        
    # docs = docs.head(500). # sample
    
    similarities = main(docs, templates)
    
    similarities = postprocess_predictions.main(similarities)
    
    # similarities.loc[:, similarities.columns != 'text'].to_csv('nDR-page-prediction/results/similarities_saved_for_postprocessing.csv')
    
    missing_pages, duplicate_pages = postprocess_alerts.main(similarities)
    
    if development_bool:
        similarities = postprocess_performance_analysis.main(similarities)

    similarities.loc[:, similarities.columns != 'text'].to_csv('nDR-page-prediction/results/similarities.csv')
    missing_pages.to_csv('nDR-page-prediction/results/missing_pages.csv')
    duplicate_pages.to_csv('nDR-page-prediction/results/duplicate_pages.csv')