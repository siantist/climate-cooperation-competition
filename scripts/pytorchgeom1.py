import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
#from torch.nn import GATConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import MessagePassing

class ricegnn(MessagePassing):
  def __init__(self, in_channels, out_channels): # ,W
      super().__init__(aggr="mean") # add, mean or max aggregation ?
      self.mlp = Sequential(
          Linear(2*in_channels, out_channels),#, weight_initializer=W),#, bias_initializer=b),
          GATConv(out_channels, out_channels) # gatconv forward( return_attention_weights = True)
      )
      self.num_features= 5
      self.hidden_size = 5
      NUM_EDGE_FEATURES = 3
      self.conv = GATConv(self.num_features, self.hidden_size, edge_dim = NUM_EDGE_FEATURES)
      self.conv2 = GATConv(self.num_features, self.hidden_size)
      self.convs = [GATConv(self.num_features, self.hidden_size, edge_dim = NUM_EDGE_FEATURES),
                      GATConv(self.hidden_size, self.hidden_size, edge_dim = NUM_EDGE_FEATURES)]

  def message(self, data):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
      #x, edge_index = data.x, data.edge_index
      #x = self.mlp(x, edge_index)
      x = self.conv2(x, edge_index, edge_attr=edge_attr)
      return x 

  
from torch_geometric.data import Data
import torch
import numpy as np
# try on random data
list_1 = np.arange(5)
list_2 = list_1
result = [[i, j] for i in list_1 for j in list_2 if i != j]
result2 = [[j, i] for i in list_1 for j in list_2 if i != j]
result3= result
result3.extend(result2)

x = torch.tensor([[-1], [0], [1], [2], [3]], dtype=torch.float)

x = np.random.rand(5,5)
x = torch.tensor(x, dtype=torch.float)
edge_index = torch.tensor(result3, dtype=torch.long)

# edge attr data with shape [num_edges, num_edge_features]
ea = torch.tensor(np.random.rand(40, 3), dtype= torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=ea)


rg = ricegnn(5,5)
rg.message(data)
