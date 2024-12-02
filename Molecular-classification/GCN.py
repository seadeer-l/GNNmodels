import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear

class _GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels,  bias=True):
        super(GraphConv, self).__init__(aggr='mean')  # 这里的聚合方式采用'mean'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight   = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_r = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        out_of_neiber = self.propagate(edge_index, x=x)
        out_of_neiber = torch.mm(out_of_neiber, self.weight)  # 线性变换

        if self.bias is not None:
            out_of_neiber = out_of_neiber +  self.bias
        
        out_of_self = torch.mm(x, self.weight_r)  # 线性变换
        out = out_of_neiber + out_of_self

        return out

    def message(self, x_j):
        return x_j

class GraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, bias=True ):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_neibors = Linear(in_channels, out_channels, bias=bias)
        self.lin_self = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_neibors.reset_parameters()
        self.lin_self.reset_parameters()

    def forward(self, x, edge_index):

        out = self.propagate(edge_index, x=x)
        out = self.lin_neibors(out)
        
        out = out + self.lin_self(x)

        return out


    def message(self, x_j):
        return x_j

    

class GCN(torch.nn.Module):
    
    def __init__(self,nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.name = 'GCN'
        self.conv1=GraphConv(nfeat,nhid)
        self.conv2=GraphConv(nhid,nhid)
        self.conv3=GraphConv(nhid,nhid)

        self.lin=Linear(nhid,nclass)

    def forward(self,x,edge_index,batch):
        x=self.conv1(x,edge_index)
        x=x.relu()	# 应用ReLU激活函数，增加模型的非线性。
        x=self.conv2(x,edge_index)
        x=x.relu()
        x=self.conv3(x,edge_index)
        x=x.relu()
        # 通过全局平均池化来聚合每个图的节点特征
        x=global_mean_pool(x,batch)
        # 应用Dropout来防止模型过拟合
        x=F.dropout(x,p=0.5,training=self.training)
        x=self.lin(x)
        return F.log_softmax(x, dim=1)
