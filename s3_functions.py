import boto3
from io import StringIO

def save_to_s3(df, path, file_name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    bucket = 'sagemaker-page-prediction-poc'
    s3_resource.Object(bucket, path + file_name).put(Body=csv_buffer.getvalue())
    print("{0} saved to {1} bucket".format(file_name, path))
    

def import_from_s3(path):
    pass