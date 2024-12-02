from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import Button, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


eColors = ["black", "blue", "green", "red"]
nColors = ['black', 'brown', 'skyblue', 'blue', 'purple', 'red', 'green']
dataset = TUDataset('data/TUDataset', name = 'MUTAG')                   


class GraphVisualizer:
    def __init__(self, root):
        self.root = root
        self.dataset = dataset
        self.current_index = 0
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.label = Label(root, text=f"Graph {self.current_index + 1}/{len(self.dataset)}")
        self.label.pack(side=tk.BOTTOM)
        
        self.next_button = Button(root, text="Next", command=self.next_graph)
        self.next_button.pack(side=tk.BOTTOM)
        
        self.prev_button = Button(root, text="PreV", command=self.prev_graph)
        self.prev_button.pack(side=tk.BOTTOM)

        self.show_graph(self.current_index)
        
    def show_graph(self, index):
        # clear screen
        self.ax.clear()        

        data = self.dataset[index]
        G = nx.Graph()
        
        # get info of gragh
        edge_index = data['edge_index'].t().cpu().numpy()
        x = data['x'].cpu().numpy()
        edge_attr = data['edge_attr'].cpu().numpy()

        # add nodes and edges into G
        G.add_edges_from(edge_index)
        
        # 根据特征设置node颜色
        node_colors = []
        for node in range(data.num_nodes):
            node_colors.insert(node,nColors[self.OneHot_to_index(x[node])])
        
        # 根据特征设置edge颜色
        edge_colors = []
        for edge in range(data.num_edges):
            edge_colors.insert(edge,eColors[self.OneHot_to_index(edge_attr[edge])])

        # 使用NetworkX绘制图，并为节点和边设置颜色
        pos = nx.spring_layout(G)  # 布局算法
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, edgelist=edge_index.tolist(), edge_color=edge_colors, alpha=0.5, width=2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')

        # 更新画布                                     
        self.canvas.draw()              
        
        # 更新标签
        self.label.config(text=f"Graph {index + 1}/{len(self.dataset)}")
        
    def next_graph(self):
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self.show_graph(self.current_index)

    def prev_graph(self):
        self.current_index = (self.current_index - 1) % len(self.dataset)
        self.show_graph(self.current_index)

    def OneHot_to_index(self, code):
        for index, element in np.ndenumerate(code):
              if element==1:
                   return index[0]




# 创建Tkinter主窗口
root = tk.Tk()
root.title("TUDataset Graph Visualizer")
 
# 初始化GraphVisualizer实例
visualizer = GraphVisualizer(root)
 
# 运行Tkinter主循环
root.mainloop()
