import pandas as pd
import time
# import python files
import store_and_load_functions


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
    path = 'predictions/'
    bucket = 'sagemaker-page-prediction-poc'
    
    # load data
    docs = store_and_load_functions.load_csv_from_s3(bucket, path, file_name = 'full_page_ocr_sample.csv')
    templates = store_and_load_functions.load_csv_from_s3(bucket, path, file_name = 'templates.csv')
    
    # preprocess data
    templates = preprocess_templates(templates)
    docs = preprocess_ocr(docs)
    
    print("preprocess_data complete in {}s".format(round(time.time() - start, 4)))
    return docs, templates


if __name__ == '__main__':
    docs, templates = main()
    print("complete")