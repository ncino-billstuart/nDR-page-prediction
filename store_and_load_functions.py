import boto3
from io import StringIO
import pandas as pd
import pickle


def save_csv_to_s3(bucket, df, path, file_name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, path + file_name).put(Body=csv_buffer.getvalue())
    print("{0} saved to {1} bucket".format(file_name, path))


def load_csv_from_s3(bucket, path, file_name):
    s3_resource = boto3.client('s3')
    obj = s3_resource.get_object(Bucket=bucket, Key=path + file_name)
    df = pd.read_csv(obj['Body'])
    print("{0} loaded from {1} bucket".format(file_name, path))
    
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

def load_from_s3(path):
    pass