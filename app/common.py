import boto3
from io import StringIO
import pandas as pd
import pickle
import logging


def define_supported_pages():
    
    supported_pages = {
       'other': ['other'],  # can have multiple
       '1120s': ['1120spage1', '1120spage3', '1120spage4', '1120spage5'],
       '1040': ['1040page1', '1040page2'], 
       'schedulec': ['schedulecpage1', 'schedulecpage2'],  # can have multiple
       'schedule': ['scheduleepage1'],  # can have multiple  # can be missing page 2
       '1065': ['1065page1', '1065page4', '1065page5'],
       'k11065': ['k11065page1'],  # can have multiple and be uploaded with 1065 or with a 1040
       '8825': ['8825page1', '8825page2'],  # can have multiple  # can be missing page 2
       '1120': ['1120page1', '1120page5', '1120page6'],  # page 6 added in 2018
       'schedule1': ['schedule1page1'],
       'schedule2': ['schedule2page1'],
       'schedule3': ['schedule3page1'],
       'schedule4': ['schedule4page1'],
                  }

    return supported_pages


def define_k1_forms():
    # purpose of idenfitying which documents a k-1 can be affiliated with
    k1_forms = ['schedulecpage1', '1040page1']
    
    return k1_forms
    

def save_csv_to_s3(bucket, df, path, file_name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, path + file_name).put(Body=csv_buffer.getvalue())
    logging.info("{0} saved to {1} bucket".format(file_name, path))


def load_csv_from_s3(bucket, path, file_name):
    s3_resource = boto3.client('s3')
    obj = s3_resource.get_object(Bucket=bucket, Key=path + file_name)
    df = pd.read_csv(obj['Body'])
    logging.info("{0} loaded from {1} bucket".format(file_name, path))
    
    return df
    
    
def store_pickle(file, path, file_name):
    # write-binary (wb) mode
    with open(path + file_name + '.pickle', 'wb') as handle:
        pickle.dump(file, handle)
      
  
def load_pickle(path, file_name): 
    # read-binary (rb) mode
    with open(path + file_name + '.pickle', 'rb') as handle:
        file = pickle.load(handle)
    
    return file