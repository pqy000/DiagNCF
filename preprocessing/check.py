#检查测试集与验证集是否有重叠，结果为否
with open("../../NCF-learn/mimic_data/mimic_val.negative", "r") as f1:
    file1 = f1.readlines()
with open("../../NCF-learn/mimic_data/mimic_test.negative", "r") as f2:
    file2 = f2.readlines()

count = 0
for i, row in enumerate(file1):
    temp1, temp2 = row.split(sep="\t")[1:-1], file2[i].split(sep="\t")[1:-1]
    for elem in temp1:
        if elem in temp2:
            print("sorry! exist")
            count+=1
print(count)
print("没有数据泄漏!")