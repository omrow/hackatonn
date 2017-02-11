import pandas as pd

def import_csv(data_path):
    data = pd.read_csv(data_path, index_col=0)
    docs = data['document']
    y = data['sentiment']
    return y, docs

def save_to_file(file_path):
    