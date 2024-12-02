from torch_geometric.datasets import TUDataset  # 从torch_geometric库中导入TUDataset类，用于加载图数据集
from torch_geometric.loader import DataLoader  # 从torch_geometric库中导入DataLoader类，用于创建数据加载器
import igraph as ig
import numpy as np

def one_hot_to_label(one_hot_encoding):
    """
    将独热编码转换为标签

    参数:
    one_hot_encoding (np.ndarray): 独热编码数组，形状为 (num_classes,)

    返回:
    int: 标签，表示激活的类的索引，如果全为0，则返回0
    """
    if np.all(one_hot_encoding == 0):
        return 0

    return np.argmax(one_hot_encoding)


def dataTreating(DatasetName, train_ratio=0.75, batch_size=64):
    """
    数据处理函数。
    
    参数:
    DatasetName (str): 数据集的名称
    train_ratio (float, 可选): 训练数据比例，默认为0.8
    batch_size  (unsigned int, 可选): 批大小，默认为64

    返回:
    包含完整数据集、训练数据加载器和测试数据加载器
    """
    
    # 加载并打乱数据集
    dataset = TUDataset('data/TUDataset', name=DatasetName)
    dataset = dataset.shuffle()
    
    # 根据比例分割数据集为训练集和测试集
    total_size = len(dataset)
    train_size = int(total_size * train_ratio) 
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    print(total_size)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # 打印训练数据加载器的信息（用于调试）
    # for step, data in enumerate(train_loader):
    #     print(f'批次 {step + 1}:')
    #     print('=======')
    #     print(f'当前批次中的图数量: {data.num_graphs}')
    #     print(data)
    #     print()
    
    return dataset, train_loader, test_loader


def dataTreating_igFormat(DatasetName, train_ratio=0.75):
    """
    将 TUDataset 数据集处理为 igraph 格式，并返回训练和测试索引。

    参数:
    DatasetName (str): 数据集的名称
    train_ratio (float, 可选): 训练数据比例，默认为0.75

  返回:
    - graphs (list[igraph.Graph]): 图的列表，每个元素是一个igraph图对象
    - node_features (list[np.ndarray]): 节点特征列表，每个元素是NumPy数组
    - labels (np.ndarray): 图的标签数组
    - train_index (np.ndarray): 训练数据的索引  
    - test_index (np.ndarray): 测试数据的索引
    """
    # 加载并打乱数据集
    dataset = TUDataset('data/TUDataset', name=DatasetName)
    dataset = dataset.shuffle()

    # 根据比例分割数据集为训练集和测试集
    total_size = len(dataset)
    train_size = int(total_size * train_ratio) 
    train_index = np.arange(train_size)  # 训练集索引
    test_index = np.arange(train_size, total_size)  # 测试集索引

    graphs = []
    node_features = []
    labels = []

    # 找到最大节点数和最大特征数
    max_num_nodes = 0
    max_num_features = 0
    
    for data in dataset:
        num_nodes = data.x.shape[0]
        num_features = data.x.shape[1]
        
        if num_nodes > max_num_nodes:
            max_num_nodes = num_nodes
        if num_features > max_num_features:
            max_num_features = num_features

    for data in dataset:
        # 提取边和节点特征
        edge_index = data.edge_index.numpy()  # 转为 NumPy 数组，形状为 (2, num_edges)
        features = data.x.numpy()  # 节点特征，形状为 (num_nodes, num_features)

        # 规整节点特征，使用零填充
        num_nodes = features.shape[0]
        padded_features = np.zeros((max_num_nodes, max_num_features))  # 创建最大大小的填充数组
        padded_features[:num_nodes, :features.shape[1]] = features
        
        lables = [one_hot_to_label(feature) for feature in padded_features]
        node_features.append(padded_features)

        # 构建 igraph 图
        edges = list(zip(edge_index[0], edge_index[1]))
        graph = ig.Graph(n=max_num_nodes, edges=edges, directed=True)
        graph.vs["label"] = lables
        graphs.append(graph) 



        # 图标签
        labels.append(data.y.item() if data.y.numel() == 1 else data.y.numpy())

    return graphs, node_features, np.array(labels), train_index, test_index


# if __name__ == "__main__":

#     # # write dataset info into txt
#     # dataset = dataTreating("MUTAG", train_ratio=0.75, batch_size=64 )
#     # write_DatasetInfo(dataset)

#     # 检查返回值
#     graphs, node_features, labels, train_index, test_index = dataTreating_igFormat("MUTAG")
#     print(f"Number of graphs: {len(graphs)}")
#     print(f"Shape of 1st graph's node features: {node_features[0].shape}")
#     print(node_features[0])
#     print(f"Shape of 2d graph's node features: {node_features[1].shape}")
#     print(f"Shape of 3d graph's node features: {node_features[2].shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Train  index: [{train_index[0]}-{train_index[-1]}]")
#     print(f"Test   index: [{test_index[0]}-{test_index[-1]}]")

