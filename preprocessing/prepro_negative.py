import pandas as pd
import numpy as np
import scipy.sparse as sp


train_data = pd.read_csv("../NCF-learn/mimic_data/mimic_train.csv", sep='\t',
                         header=None, names=['ICD9_new', 'ITEM_new', 'COUNT_new'],
                         usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
test_data = pd.read_csv("../NCF-learn/mimic_data/mimic_test.csv", sep='\t',
                        header=None, names=['ICD9_new', 'ITEM_new', 'COUNT_new'],
                        usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})
val_data = pd.read_csv("../NCF-learn/mimic_data/mimic_val.csv", sep='\t',
                       header=None, names=['ICD9_new', 'ITEM_new', 'COUNT_new'],
                       usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32})

item_num = train_data['ITEM_new'].max() + 1
user_num = train_data['ICD9_new'].max() + 1
train_data = train_data.values.tolist()
test_data = test_data.values.tolist()
val_data = val_data.values.tolist()
train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
for x in train_data:
    train_mat[x[0], x[1]] = 1.0
num_ng = 99

def test_ng_sample(target_data, name="test"):
    feature_ng = {}
    for x in target_data:
        u = x[0]
        collect_sample = []
        collect_sample.append([x[0], x[1]])
        for t in range(num_ng):
            j = np.random.randint(item_num)
            while (u,j) in train_mat:
                j = np.random.randint(item_num)
            collect_sample.append([u, j])
        feature_ng[u] = collect_sample
    write_csv(feature_ng, name)

def val_ng_sample(target_data, name="valid"):
    feature_ng = {}
    for x in target_data:
        u = x[0]
        collect_sample = []
        collect_sample.append([x[0], x[1]])
        for t in range(num_ng):
            j = np.random.randint(item_num)
            while (u,j) in train_mat:
                j = np.random.randint(item_num)
            collect_sample.append([u, j])
            train_mat[u, j] = 1.0
        feature_ng[u] = collect_sample
    write_csv(feature_ng, name)

def write_csv(source_dict, name="test"):
    with open("./mimic_data/mimic_"+name+".negative", "w") as f:
        for i, (key, value) in enumerate(source_dict.items()):
            f.write("(%d,%d)"%(key,value[0][1]))
            for item in value[1:]:
                f.write("\t"+str(item[1]))
            f.write("\n")

val_ng_sample(val_data, name="val")
test_ng_sample(test_data, name="test")







