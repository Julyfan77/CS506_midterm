import pandas as pd

# 读取 train.csv 和 test.csv
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 从 test.csv 中提取需要的 ID 列
test_ids = test['Id']

# 从 train.csv 中筛选出与 test.csv 中 ID 匹配的行
mytest = train[train['Id'].isin(test_ids)]

keyword_scores = {
    'good': 1,
    'excellent': 2,
    'bad': -1,
    'awful': -2,
    'fun': 1,
    'truth': 1,
    'intelligent': 2,
    'bittersweet': 1
}
def calculate_score(summary):
    if pd.isna(summary):
        return 0
    words = summary.lower().split()
    score = sum(keyword_scores.get(word, 0) for word in words)
    return score

def add_features_to(df):
    # This is where you can do all your feature extraction

    df['Helpfulness'] = (df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'])*10
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    df['SummaryScore'] = df['Summary'].apply(calculate_score)
    return df

# 保存为 mytest.csv
mytest.to_csv('mytest.csv', index=False)
print(f"mytest.csv has been saved with {len(mytest)} rows.")
