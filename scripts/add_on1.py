
# add on to the init part
def init():
    # sy add more properties
    self.track_values_key = {}
    self.track_actions_key = {}
    self.keys_to_optimize = ["global_temperature", "global_carbon_mass"]
    self.keys_to_maximize = ["capital_all_regions"]

    # region specific features to optimize

    # sy add a data structure to save which action led to which costs

    for key in self.keys_to_optimize:
        self.track_values_key[key] = []
        self.track_actions_key[key] = []

    for key in self.keys_to_maximize:
        self.track_values_key[key] = []
        self.track_actions_key[key] = []

    # sy add actions to the nvec that are related to investment in green energy
    self.green_energy_investment_nvec = [self.num_discrete_action_levels] * self.num_regions

    # sy add public transportation investment (high speed rail )
    self.public_transportation_investment_nvec = [self.num_discrete_action_levels] * self.num_regions
    # research and development of methods to fight climate change
    self.research_development_investment_nvec =  [self.num_discrete_action_levels] * self.num_regions
    # media

    # tree planting policies

    # map for the (i-1) actions ?
    self.previous_iteration_actions = {}

    # how to correlate the actions with the indices
    self.indices_actions ={} # starting index

    indices = 0
    self.indices_actions['savings'] = indices
    indices = indices+1
    self.indices_actions['mitigation_rate'] = indices
    indices=indices+1
    self.indices_actions['export_action'] = indices
    indices = indices+1
    self.indices_actions['import_actions'] = indices
    indices = indices+self.num_regions
    self.indices_actions['tariff_actions'] =indices
    indices = indices + 2*self.num_regions
    self.indices_actions['proposal_actions'] = indices
    indices = indices +self.num_regions
    self.indices_actions['evaluation_actions'] =indices
    indices = indices + self.num_regions
    self.indices_actions['green_energy_investment'] = indices
    indices = indices + self.num_regions
    self.indices_actions['public_transportation_investment'] = indices
    indices = indices + self.num_regions
    self.indices_actions['research_development_investment'] = indices

    self.list_actions = ['savings', 'mitigation_rate', 'export_action', 'import_actions', 'tariff_actions', 'proposal_actions', 'evaluation_actions', 'green_energy_investment', 'public_transportation_investment', 'research_development_investment']

    # phi
    num_dict= self.phi1()

    self.num_dict = num_dict
    self.relevant_corr = {} # relevant_corr

    # countries give scores to other countries based on how much attention they pay to them ( influenced by the proximity )
    attention_scores = {}
    # calculate the attention coefficients
    for i in range(self.num_regions):
        for j in range(self.num_regions):
            attention_scores[i,j] = np.random.rand()

    attention_coefficients = {}
    self.attention_coefficients = attention_coefficients

    for i in range(self.num_regions):
        for j in range(self.num_regions):
            attention_coefficients[i,j] = attention_scores[i,j]*(np.dot(np.matmul(W, h[i]), np.matmul(W, h[j])))

    #for i in range(0, sum(self.actions_nvec), 10):
        #self.indices_actions[key] = i

# add to reset

def reset():

    for key in [
        "green_energy_investment",
        "private_company_green_investment",
    ]:
        self.set_global_state(
            key=key,
            value=np.zeros((self.num_regions, self.num_regions)),
            timestep=self.timestep,
            norm=1e2)

        
# relevant action and state dictionaries
def phi4(self, actions):
    
    all_reg_action_dict = {}
    all_reg_state_dict ={}
    
    for i in range(self.num_regions):
        action_dict ={}
        state_dict = {}
        for rkey in self.relevant_states:
            rval = self.global_state[rkey]
            rlen = self.relevant_state_lens[rkey]
            if rlen==1:
                action_dict[rkey] = rval
            if 1< rlen < self.num_regions:
                action_dict[rkey] = np.mean(rval)
            if rlen == self.num_regions:
                action_dict[rkey] = rval[i]
        
        all_reg_action_dict[i] = action_dict
        
        for akey in self.relevant_actions:
            aindex = self.indices_actions[akey]
            alen = self.relevant_action_lens[akey]
            aval = actions[aindex]
            if alen==1:
                state_dict[akey] = aval
            if 1 < alen < self.num_regions:
                state_dict[akey] = np.mean(aval)
            if alen == self.num_regions:
                state_dict[akey] = aval[i]
        
        all_reg_state_dict[i] = state_dict
    
    return all_reg_action_dict, all_reg_state_dict

def run_kernel(self, g1, g2):
    from grakel.kernels import PropagationAttr
    pak = PropagationAttr(normalize=True)
    pak.fit_transform([g1])
    sc = pak.transform([g2])
    
    return sc

# actions are nodes, node attributes are action values
def prepare_grakel_input2(self, actions, all_reg_action_dict):
    from grakel import Graph
    
    for i in range(self.num_regions):
        actions_i = actions[i]
        
        node_attributes = {}
        edges = {}
        
        act_ind= 0
        # for each action, set a
        for akey in self.relevant_actions:
            
            aindex = self.indices_actions[akey]
            alen = self.relevant_action_lens[akey]
            
            # add to the node attributes
            ac1 = actions_i[aind1]
            node_attributes[act_ind] = ac1
            act_ind = act_ind+1
        
        rel_action_len = len(self.relevant_actions)
        for i in range(rel_action_len):
            ilist = np.arange(rel_action_len)
            ilist.remove(i)
            edges[i] = ilist
        ggraph = Graph(edges, node_labels= node_attributes)
        grakel_graphs.append(ggraph)
    return grakel_graphs

# add to step
def step():

    # compute the correlation
    global_correlation = self.glob_corr()
    rel_correlation = self.phi2()
    #print("relevant correlation is ", rel_correlation, "and glob correlation is", global_correlation)
    print("glob correlation is", global_correlation)

    if self.negotiation_on:
        if self.stage == 2:
            self.previous_iteration_actions = actions

def extra_step():
    # recompute the correlations
    relevant_corr = self.phi2()

# function to group the diffeerent regions given their importance preferences
def group_regions(self):

# function to group the different actions
def group_actions(self):


# create the GNN
import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
#from torch.nn import GATConv
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv import MessagePassing

def gnn1():
    class mygnn1(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr="mean") # mean or max aggregation ?
            self.mlp = Sequential(
                Linear(2*in_channels, out_channels, weight_initializer=W, bias_initializer=b),
                GATConv(out_channels, out_channels) # gatconv forward( return_attention_weights = True)
            )
        def forward(self, x:Tensor, edge_index:Tensor) -> Tensor:
            return self.propogate(edge_index, x=x)
        def message(self, x_j:Tensor, x_i:Tensor) -> Tensor:
            # x_j source node shape [num_edges, in_channels], x_i target node same shape
            edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
            return self.mlp(edge_features)

        # return the node or edge embeddings
        def get_node_embedding(self):
            return self.x

        def get_edge_embedding(self):
            return self.edge_index, self.attention_weights

def gnn2():
    class mygnn2(MessagePassing):
        def __init__(self, in_channels, out_channels):
            super().__init__(aggr='add') # add aggreg of neighboring node features
            self.lin = Linear(in_channels, out_channels, weight_initializer=W) # step 2

def train_mygnn():
    model = mygnn1(dataset.num_features, dataset.num_features) #dataset.num_classes)
    optimizer = torch.optim.Adam(params = model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad()
    ne = model.get_node_embedding()
    ee = model.get_edge_embedding()


# 2 Do : draw the weight matrix and how it should act on features

# give column name an index
def phi1(self):
    df = pd.DataFrame(self.global_state)
    column_names = list(df.columns)

    # add the column names to dictionary
    num_dict = {}
    clen = len(column_names)
    for c in range(clen):
        feat = column_names[c]
        num_dict[feat] = c

    return num_dict # , relevant_corr


def glob_corr(self):
    import pandas as pd
    df = pd.DataFrame(self.global_state)
    corr_matrix = df.corr()

    return corr_matrix

# create inverse function for features that are inversely related
def phi2(self):
    # determine the correlation (pos or neg )
    import pandas as pd
    df = pd.DataFrame(self.global_state)
    corr_matrix = df.corr()

    import numpy as np
    acorr_matrix = np.array(corr_matrix)

    relevant_corr = {}
    for key1 in self.keys_to_optimize: #self.features_dict:
        # calculate the correlation by the time series values
        for key2 in self.keys_to_optimize:
            i1 = self.num_dict[key1]
            i2 = self.num_dict[key2]
            correl = acorr_matrix[i1, i2]
            relevant_corr[(key1, key2)] =correl

    return relevant_corr

    # use this as slope of the linear function , ax + b

def prepare_grakel_input(self, actions):
    # edges = {1: {2: 0.5, 3: 0.2}, 2: {1: 0.5}, 3: {1: 0.2}}
    # node_attributes = {1: [1.2, 0.5], 2: [2.8, -0.6], 3: [0.7, 1.1]}
    # G2 = Graph(edges, node_labels=node_attributes)

    # can plug in the actions directly as node attributes
    node_att = actions

    for i in range(self.num_regions):
        ai = actions[i]
        for j in range(self.num_regions):
            # add edges that correlate
            # loop over all actions
            for a in self.list_actions:

                i1 = self.num_dict[a]

                for ckey in self.relevant_corr:
                    cval = self.relevant_corr[ckey]

        # use edges





# transform each region's action vector into node embedding as input to GNN,
# transform the proposals into edge embedding

# average over past actions,

# input to grakel kernel
def transform_input(self, actions):

    for i in range(self.num_regions):
        # look at the actions
        ai = actions[i]

        node_attributes = {1: [1.2, 0.5], 2: [2.8, -0.6], 3: [0.7, 1.1]}

        # create the node attribute for each action >

        #ai_previous = self.previous_iteration_actions[i]
        # turn
        # average over the actions

        # make a graph from it

# generate input to gnn
def generate_graph(self):
    import torch
    from torch_geometric.data import Data
    list1 = np.arange(self.num_regions)
    list2 = list1
    #result = [(i, j) for i in list_1 for j in list_2 if i != j]
    result = [[i, j] for i in list_1 for j in list_2 if i != j]
    result2 = [[j, i] for i in list_1 for j in list_2 if i != j]
    result3= result
    result3.extend(result2)
    edge_index = result3
    x = []
    for i in range(self.num_regions):
        x.append(actions[i]/10) # from the actions dictionary , normalize <= 1
    x = torch.tensor(x, dtype=torch.float)
    #[num_edges, num_edge_features]
    edge_feature_matrix = np.zeros((nedges, nfeatures))
    # only take the edge features from the relevant features

    data = Data(x=x, edge_index = edge_index, edge_attr= edge_feature_matrix)


    # write to pickle
    pickle_out = open('datagraph.pickle', 'wb')
    pickle.dump(const_map, pickle_out)
    pickle_out.close()


def train_gnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


# function to convert the weight into a sampled action matrix ?
# based on def generate_action_masks
def sample_weights_action_mask(self, wts_dict, num_regions, relevant_actions, action_index):
  mask_dict = {region_id: None for region_id in range(self.num_regions)}

  mask_start = 1164
  mask_end = mask_start + 10
  # locate the relevant actions
  for ra in relevant_actions:
      mask_start = action_index[ra]
      mask_end = mask_start + 10

      for i in range(self.num_regions):
        wts_i = wts_dict[i]
        #sample each action
        mask = self.default_agent_action_mask.copy()

        if self.negotiation_on:
          # sample the action
          action_array = []
          for j in range(self.len_actions):
              action01 = np.random.choice([1,0], p =[wts_i[j], 1- wts_i[j]])
              action_array.append(action01)

        # build the mask from this action
          mask_dict[mask_start:mask_end] = action_array



# relate the new features?
def extra_simulation_step(self, actions=None):
    # relate the new features to existing features and update global state
