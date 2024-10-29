import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler
trainingSet = pd.read_csv("./train.csv")[:19000]
testingSet = pd.read_csv("./mytest.csv")

print("train.csv shape is ", trainingSet.shape)
print("test.csv shape is ", testingSet.shape)

print(trainingSet.head())
print()
print(testingSet.head())


print(trainingSet.describe())



keyword_scores = {
    'good': 1,
    'excellent': 2,
    'bad': -1,
    'terrible':-1,
    'awful': -2,
    'fun': 1,
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
    df['SummaryScore'] = df['Summary'].apply(calculate_score)*0
    #print(df['SummaryScore'][:5])
    return df

# Load the feature extracted files if they've already been generated

# Process the DataFrame
train = add_features_to(trainingSet)
test = add_features_to(testingSet)
scaler = StandardScaler()

# 选择需要标准化的列（假设是 'Feature1' 和 'Feature2'）
features = ['Helpfulness', 'Time']

# 对特征进行标准化
train[features] = scaler.fit_transform(train[features])
test[features]= scaler.fit_transform(test[features])
# Merge on Id so that the submission set can have feature columns as well
X_submission = test

# The training set is where the score is not null
X_train =  train[train['Score'].notnull()]
X_train.to_csv("./X_train.csv", index=False)

features = [ 'Time','HelpfulnessDenominator','SummaryScore' ,'UserId','ProductId','Helpfulness']

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
    X_train.drop(columns=['Score']),
    X_train['Score'],
    test_size=1/1000.0,
    random_state=0
)

X_train_select = X_train[features]
X_submission_select = X_submission[features]
print(X_submission_select)
import sys
sys.stdout.flush() 
import KNN
import importlib
import numpy as np
importlib.reload(KNN)
model=KNN.CustomKNN(n_neighbors=38,use_weights="helpful")
model.fit(X_train_select, Y_train)





print(X_submission_select.shape)
batch_size = 1000
predictions = np.zeros(len(X_submission_select))
num_batches = (len(X_submission_select) + batch_size - 1) // batch_size  # 计算批次数
for i in range(num_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(X_submission_select))
    
    batch = X_submission_select.iloc[start:end]
    batch_predictions = model.predict(batch)
    
    # 将批次预测结果存入对应位置
    predictions[start:end] = batch_predictions
    print(str(i*1000)," predicted")
    sys.stdout.flush() 
X_submission['Score'] = predictions
submission = X_submission[['Id', 'Score']]
submission.to_csv("./submissionk=38_num=16000_wordsummary.csv", index=False)