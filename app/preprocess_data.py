import pandas as pd
import time
import logging
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# local app imports
import app.common as common


def preprocess_templates(templates):
    
    templates = templates.sort_values(by='type_year').reset_index(drop=True)
    templates['type'] = templates['type'].str.lower()
    
    return templates
    
    
def preprocess_ocr(docs):
    
    keep_columns = ['file_name', 'year_file_name', 'page_results', 'target', 'text']
    docs = docs[keep_columns]
    docs = docs.rename(columns={'year_file_name': 'year', 'page_results': 'page'})
    
    # remove pages with no OCR (no recognized text on page)
    len_before = len(docs)
    docs = docs[docs['text'].notna()]
    if len_before - len(docs) > 0:
        print("warning: {} pages removed due to no OCR recognized text".format(len_before - len(docs)))
    
    # remove some years as 2019 is fake data
    docs = docs[docs['year'].isin([2015, 2016, 2017, 2018])]

    return docs


def main():
    
    start = time.time()
    path = 'full_page_ocr/gulfcoast-test-data/results/'
    bucket = 'ncino-mlplatform-data-protected'
    
    # load data
    s3_resource, s3_client = common.assume_data_role()
    docs = common.load_csv_from_s3(s3_client, bucket, path, file_name = 'ocr_simple.csv')
    templates = common.load_csv_from_s3(s3_client, bucket, path, file_name = 'templates.csv')
    
    # preprocess data
    templates = preprocess_templates(templates)
    docs = preprocess_ocr(docs)
    
    logging.info("{}s".format(round(time.time() - start, 4)))
    return docs, templates


if __name__ == '__main__':
    
    docs, templates = main()