import os.path, os
import urllib.request
import zipfile

def download_data():
    if not os.path.exists('../data/aggression_annotations.tsv'):
        
        # file_path = '../data/4054689.zip'
        # urllib.request.urlretrieve('https://ndownloader.figshare.com/articles/4054689/versions/6', file_path)
        # with zipfile.ZipFile(file_path, 'r') as zip_ref:
        #     zip_ref.extractall('../data')

        # os.remove(file_path)
        
        file_path = '../data/4267550.zip'
        urllib.request.urlretrieve('https://ndownloader.figshare.com/articles/4267550/versions/5', file_path)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('../data')

        os.remove(file_path)

        # file_path = '../data/4563973.zip'
        # urllib.request.urlretrieve('https://ndownloader.figshare.com/articles/4563973/versions/2', file_path)
        # with zipfile.ZipFile(file_path, 'r') as zip_ref:
        #     zip_ref.extractall('../data')

        # os.remove(file_path)
