import pandas as pd

# 读取 train.csv 和 test.csv
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 从 test.csv 中提取需要的 ID 列
test_ids = test['Id']

# 从 train.csv 中筛选出与 test.csv 中 ID 匹配的行
mytest = train[train['Id'].isin(test_ids)]

# 保存为 mytest.csv
mytest.to_csv('mytest.csv', index=False)

print(f"mytest.csv has been saved with {len(mytest)} rows.")
