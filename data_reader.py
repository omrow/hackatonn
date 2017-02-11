import pandas as pd

def import_csv(data_path):
    data = pd.read_csv(data_path, index_col=0)
    docs = data['document']
    y = data['sentiment']
    return y, docs

def save_data_to_txt_file(filename, labels, data):
    file = open(filename, 'w')
    for i in range(0, len(labels)):
        file.write("%s\n" % labels[i])
        file.write("%s\n" % data[i])

def read_data_from_txt_file(filname):
    with open(filname, "r") as dats:
        ys = []
        docs = []
        for i, line in enumerate(dats):
            if i % 2 == 0:
                ys.append(line)
            else:
                docs.append(line)
        return ys, docs

def read_test_data(path):
    with open(path, 'r') as post_table:
        posts = post_table.readlines()
        posts = [line.split('|') for line in posts]
        posts = [sub_list[4] for sub_list in posts]
    return posts