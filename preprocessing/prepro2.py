import pandas as pd
import json

f = open("ICD_DICT.json")
data_icd = json.load(f)
for key, value in enumerate(data_icd):
    data_icd[value] = key
with open("ICD_DICT_number.json", "w") as outfile:
    json.dump(data_icd, outfile, indent=2)

f = open("LABITEM_DICT.json")
data_lab = json.load(f)
for key, value in enumerate(data_lab):
    data_lab[value] = key
with open("LABITEM_DICT_number.json", "w") as outfile:
    json.dump(data_lab, outfile, indent=2)

diagnoses_data = pd.read_csv("DIAGNOSES_ICD.csv", sep=",")
labevents_data = pd.read_csv("LABEVENTS.csv", sep=",")
diagnoses_data["ICD9_new"] = diagnoses_data["ICD9_CODE"].replace(data_icd)
labevents_data["ITEMID"] = labevents_data["ITEMID"].astype(str)
labevents_data["ITEMID_new"] = labevents_data["ITEMID"].replace(data_lab)

# print(diagnoses_data["ICD9_CODE"].nunique())
# print(labevents_data["ITEMID"].nunique())

diag_data  = diagnoses_data.loc[diagnoses_data["SEQ_NUM"]==1]
merge_table = pd.merge(diag_data, labevents_data, on="HADM_ID")
id_num = merge_table["SUBJECT_ID_x"].nunique()
hadm_num = merge_table["HADM_ID"].nunique()

# 表明有些病症未作为主要住院的原因
# print(merge_table["ICD9_CODE"].nunique())     #934
# print(merge_table["ITEMID"].nunique())  #710

merge_table = merge_table.drop(["SUBJECT_ID_y"], axis=1)
merge_table = merge_table.drop(["SEQ_NUM"], axis=1)
merge_table["COUNT_new"] = merge_table["ITEM_COUNT"]
result = merge_table.loc[merge_table["FLAG"]==1]

result.to_csv("merge_table.csv")
sorted_result = result.sort_values(by=['ICD9_new'], ascending=True)
sorted_result.to_csv("merge_table_sorted.csv")
