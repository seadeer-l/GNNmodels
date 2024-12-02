from wwl import wwl
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from utils import dataTreating_igFormat
from sklearn.model_selection import GridSearchCV

class WL_SVM_train():
    def __init__(self, DatasetName, train_ratio, iter):
        # 加载数据集
        self.graphs, self.node_features, self.labels, self.train_index, self.test_index = dataTreating_igFormat(DatasetName, train_ratio=train_ratio)
        self.iter = iter
        self.compute_wwlKernel()
        
    def compute_wwlKernel(self):
        # 计算核矩阵：调用 wwl 函数计算核矩阵
        self.kernel_matrix = wwl(self.graphs, node_features=self.node_features, num_iterations=self.iter)

    def train(self):
        # 提供训练样本之间的相似性，用于 SVM 的训练。
        self.K_train = self.kernel_matrix[self.train_index][:, self.train_index]
        # 提供测试样本与训练样本之间的相似性，用于 SVM 的预测。
        self.K_test = self.kernel_matrix[self.test_index][:, self.train_index]
        # 初始化 SVM 模型
        self.svm = SVC(kernel='precomputed')
        # 使用网格搜索进行超参数调优
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],  # 惩罚参数
            'gamma': ['scale', 'auto']  # 核函数参数
        }
        grid_search = GridSearchCV(self.svm, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.K_train, self.labels[self.train_index])  # 在训练数据上拟合
        # 输出最佳参数
        # print("Best parameters found: ", grid_search.best_params_)
        # 使用最佳参数训练 SVM
        best_svm = grid_search.best_estimator_

        # 进行预测
        y_predict_train = best_svm.predict(self.K_train)
        y_predict_test = best_svm.predict(self.K_test)

        # 输出预测结果
        # print(classification_report(self.labels[self.test_index], y_predict))
        string = f'Model: WL_SVM, Train Acc: {accuracy_score(self.labels[self.train_index], y_predict_train):.4f}, Test Acc: {accuracy_score(self.labels[self.test_index], y_predict_test):.4f}'
        print(string)
        filename = 'output.txt'
        with open(filename, 'a', encoding='utf-8') as file: file.write(string+'\n')


















