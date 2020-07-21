import pandas as pd

import s3_functions


def preprocess_templates(templates):
    templates = templates.sort_values(by='type_year').reset_index(drop=True)
    templates['type'] = templates['type'].str.lower()
    
    return templates
    
def preprocess_ocr():


if __name__ == '__main__':
    path = 'predictions/'
    bucket = 'sagemaker-page-prediction-poc'
    
    docs = s3_functions.load_csv_from_s3(bucket, path, file_name = 'full_page_ocr_sample.csv')
    templates = s3_functions.load_csv_from_s3(bucket, path, file_name = 'templates.csv')
    
    templates = preprocess_templates(templates)
    
    print("hello")