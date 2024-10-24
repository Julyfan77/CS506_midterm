import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, n_neighbors=3, use_weights=None):
        """
        n_neighbors: 邻居数量
        use_weights: 自定义的权重列（传入数组或 None）
        """
        self.n_neighbors = n_neighbors
        self.use_weights = use_weights
        self.pindex=-2
        self.hindex=-1
        print("init")
        #w={}
    def fit(self, X, y):
        """保存训练数据"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.weights_column = self.X_train[:, self.hindex]
        self.pids=self.X_train[:, self.pindex]
        
        self.X_train=np.delete(self.X_train, self.pindex, axis=1)
        print("fit",self.X_train.shape)
        return self
    
    def _euclidean_distance(self, x1, x2):
        """计算两个样本之间的欧几里得距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
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
    def _get_neighbors(self, x, x_pid):
    
        # 1. 先找出 pid 匹配的邻居
        matching_neighbors = [
            (self._euclidean_distance(x, train_sample), label, i)
            for i, (train_sample, label, pid) in enumerate(zip(self.X_train, self.y_train, self.pids))
            if pid == x_pid
        ]
        
        # 2. 按照距离排序
        sorted_matching = sorted(matching_neighbors, key=lambda x: x[0])
        #print("id match dis",sorted_matching)
        
        # 3. 如果匹配的邻居数量足够，直接返回
        if len(sorted_matching) >= self.n_neighbors:
            return sorted_matching[:self.n_neighbors]

        # 4. 若不足 n_neighbors，则找出所有其他非匹配的邻居
        non_matching_neighbors = [
            (self._euclidean_distance(x, train_sample), label, i)
            for i, (train_sample, label, pid) in enumerate(zip(self.X_train, self.y_train, self.pids))
            if pid != x_pid
        ]
        
        # 5. 按距离排序非匹配邻居
        sorted_non_matching = sorted(non_matching_neighbors, key=lambda x: x[0])

        # 6. 补齐缺少的邻居
        needed = self.n_neighbors - len(sorted_matching)
        additional_neighbors = sorted_non_matching[:needed]
        #print("unmathced dis",additional_neighbors)
        # 7. 合并匹配和补充的邻居，并返回
        final_neighbors = sorted_matching + additional_neighbors

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
        print(f"Original X shape: {X.shape}")

        # 提取 PID 列，并确保是正确的轴切片
        x_pids = X[:, self.pindex]  # 提取 PID 列
        print(f"PIDs shape after extraction: {x_pids.shape}")

        # 删除 PID 列，只保留特征
        X = np.delete(X, self.pindex, axis=1)
        print(f"Features shape after deleting PID column: {X.shape}")
        
        for sample, xpid in zip(X,x_pids):
            neighbors = self._get_neighbors(sample,xpid)
            pred = self._weighted_vote(neighbors)
            
            predictions.append(pred)
        print(len(predictions))
        return np.array(predictions)
