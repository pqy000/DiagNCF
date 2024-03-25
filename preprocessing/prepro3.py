import pandas as pd
import numpy as np
import scipy.sparse as sp

overall_data = pd.read_csv("merge_table_sorted.csv", sep=",",
            names=['ICD9_new', 'ITEM_new', 'COUNT_new'],
            usecols=[4, 8, 9], header=0)
print(overall_data["ICD9_new"].nunique())
grouped = overall_data.groupby(["ICD9_new"])
size = grouped.size()
min_size = 30

# shuffle in each group
# grouped = grouped.apply(lambda x: x.sample(frac=0.8)).reset_index(drop=False)
# temp2 = grouped.groupby(["ICD9_new"]).apply(lambda x: x)
# temp2 = temp2.apply(lambda x: x)
# temp = grouped.apply(lambda x: x)

def check_min_grouped(min_size = 25):
    # 查看不同group size大小
    group_sizes, group_other_sizes = {}, {}
    valid_groups = []
    min_count = 0
    for key, group in grouped:
        if(len(group) > min_size):
            valid_groups.append(group)
            group_sizes[key] = len(group)
        else:
            group_other_sizes[key] = len(group)
            min_count += 1
    return min_count, len(group_sizes)

min_count, valid_count = check_min_grouped(min_size)
print(min_count, valid_count)
use_grouped = grouped.filter(lambda x: x.shape[0] >= min_size)
other_grouped = grouped.filter(lambda x: x.shape[0] < min_size)
print(use_grouped.size, other_grouped.size)
temp = use_grouped.groupby(["ICD9_new", "ITEM_new"]).size().reset_index(name='COUNT_new')
print(len(temp))
print("Item_size")
print(use_grouped['ITEM_new'].nunique())
use_grouped = use_grouped.groupby(["ICD9_new"])
# use_grouped = temp.grouped.groupby(["ICD9_new"])


# separate train/test set
train_data = use_grouped.apply(lambda x: x.iloc[:-3])
valid_data = use_grouped.apply(lambda x: x.iloc[-2])
test_data = use_grouped.apply(lambda x: x.iloc[-1])

train_data = train_data.apply(lambda x: x)
valid_data = valid_data.apply(lambda x: x)
test_data = test_data.apply(lambda x: x)

train_data.to_csv("./mimic_data/mimic_train.csv", index=False, header=False, sep='\t')
valid_data.to_csv("./mimic_data/mimic_val.csv", index=False, header=False, sep='\t')
test_data.to_csv("./mimic_data/mimic_test.csv", index=False, header=False, sep='\t')








