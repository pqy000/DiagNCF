# dataset name 
dataset = 'mimic'
assert dataset in ['ml-1m', 'pinterest-20', 'mimic']

# model name 
model = 'NeuMF-end'

assert model in ['GMF', 'MLP', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = '/home/qingyi/GPU33/test/NCF-learn/'
min_value = 30
data_path = "mimic_data_"+str(min_value)+"/"

train_rating = main_path + data_path +'{}_train.csv'.format(dataset)
test_rating = main_path + data_path + '{}_test.csv'.format(dataset)
test_negative = main_path + data_path + '{}_test.negative'.format(dataset)
val_rating = main_path + data_path + '{}_val.csv'.format(dataset)
val_negative = main_path + data_path + '{}_val.negative'.format(dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
