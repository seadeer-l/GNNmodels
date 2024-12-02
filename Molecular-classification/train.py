from GCN import GCN
from GIN import GIN
from SVM import WL_SVM_train
import argparse
import torch
import utils


class train():
    def __init__(self,model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion


    def result(self,train_loader, test_loader, epochs):    
        for epoch in range(epochs): # T为训练的周期数
            self.train(train_loader)
            self.train_acc=self.evaluate(train_loader)
            self.test_acc=self.evaluate(test_loader)
            string = f'Model: {model.name}, Epoch: {epoch+1:03d}, Train Acc: {self.train_acc:.4f}, Test Acc: {self.test_acc:.4f}'
            print(string)
            # 文件名
            filename = 'output.txt'
            with open(filename, 'a', encoding='utf-8') as file:
                file.write(string+'\n')



    # 训练
    def train(self, train_loader):
        self.model.train()  # 将模型设置为训练模式
        for traindata in train_loader:  # 遍历训练数据加载器中的每个批次
            # 前向传播：将数据（节点特征、边索引和批次信息）传递给模型
            out = self.model(traindata.x, traindata.edge_index, traindata.batch)
            # 计算损失：使用输出和真实标签来计算交叉熵损失
            loss = self.criterion(out, traindata.y)
            # 反向传播：计算损失相对于模型参数的梯度
            loss.backward()
            # 参数更新：使用优化器更新模型参数
            self.optimizer.step()
            # 优化器梯度清零
            self.optimizer.zero_grad()
            
    # 测试
    def evaluate(self, test_loader):
        self.model.eval()  # 将模型设置为评估模式
        correct = 0
        total = 0  # 添加一个变量来跟踪总样本数
        for testdata in test_loader :
            out = self.model(testdata.x, testdata.edge_index, testdata.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == testdata.y).sum().item())  # 使用.item()将张量转换为Python标量
            total += testdata.y.size(0)  # 假设data.y包含了当前批次中所有样本的标签，因此其第一个维度的大小就是批次大小
        accuracy = correct / total  # 计算准确率
        return accuracy
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model to train.')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='Model to train.')
    parser.add_argument('--tr', type=float, default=0.75,
                        help='Train ratio of dataset')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--iter', type=int, default=1,
                        help='Number of WL kernel\' iteration')
    args = parser.parse_args()

   # 加载和处理数据
    dataset,train_loader,test_loader = utils.dataTreating(DatasetName=args.dataset, train_ratio=0.75, batch_size=64)

    # 提取数据集的输入维度和类别数量
    input_dim = dataset.num_features
    output_dim = dataset.num_classes
    hidden_dim = args.hidden


    if args.model == 'SVM':
        WL_SVM_train(DatasetName=args.dataset, train_ratio=args.tr, iter=args.iter ).train()
    else:
        if args.model == 'GIN':
            model = GIN(input_dim, hidden_dim, output_dim)
        elif args.model == 'GCN':
            model = GCN(input_dim, hidden_dim, output_dim)

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        # 初始化训练器
        trainer = train(model, optimizer, criterion)

        # 开始训练和评估
        trainer.result(train_loader, test_loader, epochs = args.epochs)

    # print(args)