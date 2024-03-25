# DiagNCF
### A pytorch GPU implementation of #016095 al. "Diagnosis Neural Collaborative Filtering for Accurate Medical Recommendation" of Qualification.


## The requirements are as follows:
	* python==3.6
	* pandas==0.24.2
	* numpy==1.16.2
	* pytorch==1.0.1
	* gensim==3.7.1
	* tensorboardX==1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)

## Example to run:
```
python main.py
```

### Dataset
We provide two processed datasets: MIMIC3-20 and MIMIC3-30 

train.rating: 
- Train file.
- Each Line is a training instance: ICD9_new\t ITEM_new\t COUNT_new (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: ICD9_new\t ITEM_new\t COUNT_new (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (ICD9_new,ITEM_new)\t negativeItemID1\t negativeItemID2 ...

### Pretrained weights
The trained model weights are under "models/"