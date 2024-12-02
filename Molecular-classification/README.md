# 图分类模型

## 环境

```py
python==3.8
torch==2.3.0
networkx==3.1
numpy==1.24.3
torch-geometric==2.6.1
matplotlib==3.7.5
wwl==0.1.2
scikit-learn
igraph
```

## 如何run？

```
path:

├── GCN.py
├── GIN.py
├── SVM.py
├── train.py
├── utils.py
├── visual.py
└── data
    └── TUDataset
        ├── BZR
        ├── COX2
        └── MUTAG
```

文件结构如上，其中：

- `data`包含了所有的数据集，本项目通过torch-geometric直接加载TUDataset数据库中的数据集。

- `GCN.py`,`GIN.py`,`SVM.py`分别实现了三种分类模型。

- `utils.py`是工具包模块，主要用于加载数据。

- `visual.py`提供一个简单的可视化窗口，让使用者直观地看到模型使用的数据，默认加载MUTAG数据集。在`path`目录下命令行输入以下命令

  ```
  python visual.py
  ```

- `train.py`用于实验训练模型，调用三个模型模块，和工具模块。参数如下：

  - `--model`：选择训练模型，有`GCN`，`GIN`,`SVM`可选。

  - `--dataset`：选择数据集，可自行访问[TDUdatabase](https://chrsmrrs.github.io/datasets/ )官网查找可用数据集。

  - `--tr`：训练集占比

  - `--epochs`神经网络训练周期，SVM模型不需要这个参数

  - `--hidden`神经网络隐藏层嵌入维度，SVM模型不需要这个参数

  默认参数为：
  
  ```python
  args(dataset='MUTAG', epochs=400, hidden=16, model='GCN', tr=0.75)
  ```