import boto3
import pandas as pd
from functools import reduce

try:
    from StringIO import StringIO #python2
except:
    from io import StringIO #python3


def make_s3_resource():
    s3 = boto3.resource('s3',
            # aws_access_key_id=os.getenv('S3_BLOG_UPLOAD_ACCESS_KEY'),
            # aws_secret_access_key=os.getenv('S3_BLOG_UPLOAD_SECRET'),
            region_name='us-east-1')
    return s3


def write_s3_file(bucket_name, s3_filename, content):
    s3conn = make_s3_resource()
    s3conn.Object(bucket_name, s3_filename).put(
            Body=content)


def read_s3_file(bucket_name, s3_filename):
    s3conn = make_s3_resource()
    # try:
    return s3conn.Object(bucket_name, s3_filename).get()["Body"].read()
    # except botocore.exceptions.ClientError as e:


def s3_csv_to_df(bucket_name, s3_filename):
    blah = read_s3_file(bucket_name, s3_filename)
    foo = StringIO(blah.decode("utf-8"))
    return pd.read_csv(foo)


def big_s3_csv_to_df(bucket_name, s3_filename_prefix, suffixes):
    filenames = [s3_filename_prefix + suff
            for suff in suffixes]
    # return filenames
    parts = [read_s3_file(bucket_name, s3_filename) 
            for s3_filename in filenames ]
    blah = reduce(lambda x, y: x+y, parts)
    foo = StringIO(blah.decode("utf-8"))
    return pd.read_csv(foo)


def df_to_s3(bucket_name, df, s3fn, index=False):
    s = StringIO()
    df.to_csv(s, index=index)
    write_s3_file(bucket_name, s3fn, content=s.getvalue())

