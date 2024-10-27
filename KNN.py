import numpy as np
from collections import Counter
import sys

class CustomKNN:
    def __init__(self, n_neighbors=3, use_weights=None):
        """
        n_neighbors: 邻居数量
        use_weights: 自定义的权重列（传入数组或 None）
        """
        self.n_neighbors = n_neighbors
        self.use_weights = use_weights
        self.uindex=-3
        self.pindex=-2
        self.hindex=-1
        self.matchedId=[]
        print("init")
        #w={}
    def fit(self, X, y):
        """保存训练数据"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.weights_column = self.X_train[:, self.hindex]
        self.pids=self.X_train[:, self.pindex]
        self.uids=self.X_train[:, self.uindex]
        self.X_train=np.delete(self.X_train, self.uindex, axis=1)
        self.X_train=np.delete(self.X_train, self.pindex, axis=1)
        print("fit",self.X_train.shape)
        return self
    
    def _euclidean_distance(self, x1, x2):
        """计算两个样本之间的欧几里得距离"""
        try:
            return np.sqrt(np.sum((x1 - x2) ** 2))
        except:
            print(x1,x2)
    
    # def _get_neighbors(self, x,x_pid):
        
    #     """找到与样本 x 最近的 n_neighbors 个邻居"""
    #     distances = [
    #     (self._euclidean_distance(x, train_sample), label, i)
    #     for i, (train_sample, label, pid) in enumerate(zip(self.X_train, self.y_train, self.pids))
    #     #if pid == x_pid  # 只保留 pid 匹配的邻居
    #     ]
    #     # 按照距离从小到大排序
    #     sorted_distances = sorted(distances, key=lambda x: x[0])
    #     # 取前 n_neighbors 个邻居
    #     #print(sorted_distances)
    #     return sorted_distances[:self.n_neighbors]
    def _get_neighbors(self, x, x_pid, x_uid):
    # 1. 找出 pid 匹配的邻居
        pid_matching_neighbors = [
            (self._euclidean_distance(x, train_sample), label, i)
            for i, (train_sample, label, pid, uid) in enumerate(zip(self.X_train, self.y_train, self.pids, self.uids))
            if pid == x_pid
        ]

        # 2. 按照距离排序 pid 匹配的邻居
        sorted_pid_matching = sorted(pid_matching_neighbors, key=lambda x: x[0])
        self.matchedId.append(len(sorted_pid_matching))

        # 3. 如果匹配的邻居数量足够，直接返回
        if len(sorted_pid_matching) >= self.n_neighbors:
            return sorted_pid_matching[:self.n_neighbors]

        # 4. 如果 pid 匹配不够，继续找 uid 匹配的邻居（但 pid 不匹配）
        needed = self.n_neighbors - len(sorted_pid_matching)
        uid_matching_neighbors = [
            (self._euclidean_distance(x, train_sample), label, i)
            for i, (train_sample, label, pid, uid) in enumerate(zip(self.X_train, self.y_train, self.pids, self.uids))
            if pid != x_pid and uid == x_uid
        ]

        # 5. 按距离排序 uid 匹配的邻居
        sorted_uid_matching = sorted(uid_matching_neighbors, key=lambda x: x[0])
        #sorted_uid_matching=[]
        # 6. 如果 pid + uid 匹配的邻居数量足够，返回它们
        if len(sorted_uid_matching) >= needed:
            final_neighbors = sorted_pid_matching + sorted_uid_matching[:needed]
            return final_neighbors

        # 7. 如果 pid 和 uid 都不足，从其他非匹配邻居中补充
        remaining_needed = needed - len(sorted_uid_matching)
        non_matching_neighbors = [
            (self._euclidean_distance(x, train_sample), label, i)
            for i, (train_sample, label, pid, uid) in enumerate(zip(self.X_train, self.y_train, self.pids, self.uids))
            if pid != x_pid and uid != x_uid
        ]

        # 8. 按距离排序其他非匹配邻居
        sorted_non_matching = sorted(non_matching_neighbors, key=lambda x: x[0])

        # 9. 合并所有邻居，补齐缺失部分
        additional_neighbors = sorted_uid_matching + sorted_non_matching[:remaining_needed]
        final_neighbors = sorted_pid_matching + additional_neighbors

        return final_neighbors

    
    def _weighted_vote(self, neighbors):
        """基于邻居的权重进行加权投票"""
        total_weight = 0
        weighted_sum = 0
        if self.use_weights == "helpful":
            weighted_votes = {}
            for dist, label, idx in neighbors:
                # 使用邻居的权重列值作为权重
                weight = self.weights_column[idx]
                weighted_score = (weight/4 + 1 )
                weighted_sum += weighted_score * label
                total_weight += weighted_score
                weighted_votes[label] = weighted_votes.get(label, 0) + weighted_score
            # prediction = round(weighted_sum / total_weight)
            # return prediction
            # 返回加权得分最高的标签
            
            return max(weighted_votes, key=weighted_votes.get)

        else:
            weighted_votes = {}
            for dist, label, idx in neighbors:
                weight = 1
                weighted_score = 1
                weighted_votes[label] = weighted_votes.get(label, 0) + weighted_score
        return max(weighted_votes, key=weighted_votes.get)
        # 返回得票最高的标签
        

    def predict(self, X):
        """对新数据进行预测"""
        predictions = []
        X=np.array(X)
        

        # 提取 PID 列，并确保是正确的轴切片
        x_pids = X[:, self.pindex]  # 提取 PID 列

        # 删除 PID 列，只保留特征
        
        x_uids = X[:, self.uindex] 
        X = np.delete(X, self.uindex, axis=1)
        X = np.delete(X, self.pindex, axis=1)
        cnt=0
        for sample, xpid,xuid in zip(X,x_pids,x_uids):
            cnt+=1
            neighbors = self._get_neighbors(sample,xpid,xuid)
            pred = self._weighted_vote(neighbors)
            # if(cnt%1000==0):
            #     print(cnt,"already predicted")
            predictions.append(pred)
        # print(self.matchedId,self.matchedId)
       
        sys.stdout.flush() 
        return np.array(predictions)
