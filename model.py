import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import copy
from torch.autograd import Variable
from GAT import GraphAttentionLayer
from dfg import DFGFeatures
from adg import ADGFeatures
from loadData import getDfgAdj, getAdgAdj
from visual import GraphVisual
from utils import *

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)

        x = F.dropout(x, self.dropout, self.training)

        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    



""" MCT Node definition"""
class MCTNode:
    def __init__(self, prob,distance, parent=None):
        # self.state = state
        self.parent = parent 
        self.children = {}
        self.visits = 0
        self.value = 0
        self.p = prob
        self.u = 0
        self.q = 0
        self.h = 1/distance # heuristic value
        self.policy = None

    def is_fully_expanded(self, state):

        return len(self.children) >= len(state.get_actions()) or self.children == {}

    def is_leaf(self):
        return self.children == {}
    
    def best_child(self, c_param=5):
        # choices_weights = [
        #     (child.q / (child.visits)) + c_param * self.p * (math.sqrt(self.visits) / (1 + child.visits)) for act, child in self.children.items()
        #     if child.visits > 0
        # ]
        # choices_act = [
        #     act for act, child in self.children.items()
        # ]

        return max(self.children.items(), key=lambda x: x[1].get_value(c_param))
    
    def get_value(self, c_param):
        self.u = c_param * self.p * math.sqrt(self.parent.visits) / (1 + self.visits)
        # print("u is {}".format(self.u))
        # print("q is {}".format(self.q))
        # print("h is {}".format(self.h))
        return self.q + self.u + 50*self.h



    def expand(self, action_probs, state):
        # tried_actions = [child.state.last_action for action, child in self.children.items()]
        legal_actions = state.get_actions()
        for action, prob in action_probs:
            if action not in self.children and action in legal_actions:
                # new_state = state.take_action(action)
                distance = state.get_distance(action)
                child = MCTNode(prob, distance, self)
                self.children[action] = child
    
    def simulate(self, state):
        current_state = copy.deepcopy(state)
        while not current_state.is_terminal():
            current_state.take_ramdom_action()
        ## print("sim finish")
        return current_state.get_reward()
    
    def backpropagate(self, result):
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
        self.q += 1.0*(result - self.q) / self.visits


""" MCT definition"""
class MCTS:
    def __init__(self, agent_net):
        self.root = MCTNode(1.0, 1)
        self.agent_net = agent_net
        # self.agent_net = 
        self.exploration_weight = 1.4

    def search(self, state):
        
        act, node = self.select(state)
        action_probs, value = self.agent_net.policy_value_fn(state)
        node.policy = action_probs
        if not state.is_terminal():
            node.expand(action_probs, state)
        else:
            value = state.get_reward()
        node.backpropagate(value)
    
    def select(self, state):
        node = self.root
        act = None
        while not node.is_leaf() and not state.is_terminal():
            act, node = node.best_child()
            state.take_action(act)
        return act, node
    
    def get_move_probs(self, state, iters, temp=1e-3):
        for i in range(iters):
            state_cp = copy.deepcopy(state)
            self.search(state_cp)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        visits = np.array(visits)
        act_probs = softmax(1.0/temp * np.log(visits+1))
        return acts, act_probs
    
    def update_with_move(self, last_move):
        """update mcts root with last move"""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None 
        else:
            self.root = MCTNode(1.0,1)

    
class Environment:
    def __init__(self, dfg, adg, operations):

        self.mask = np.zeros((len(dfg.getNodes()), adg.getMaxNodeId()+1))
        self.dfg_mask = np.zeros((len(dfg.getNodes()), adg.getMaxNodeId()+1))
        self.dfg = dfg 
        self.adg = adg
        self.ops = operations
        self.dfgFeatures = DFGFeatures(dfg)
        self.dfg_features = self.dfgFeatures.features

        self.dfg_features_vec = self.dfgFeatures.getFeaturesVector(operations)
        self.adgFeatures = ADGFeatures(adg)
        self.adg_features = self.adgFeatures.features
        self.adg_features_vec = self.adgFeatures.getFeaturesVector()
        self.dfg_adj = getDfgAdj(dfg)
        self.adg_adj = getAdgAdj(adg)
        self.init_env = copy.deepcopy((dfg, adg, operations))
        self.reward = {}
        self.adg_actions = self.adg.getMaxNodeId() + 1
        self.graph_visual = GraphVisual(dfg,adg)
        self.init_mask()

    def update(self, action, reward, print_action = False):
        if (self.canMapping(action, reward)):
            self.updateFeatures(action)
            self.updateAdj(action)
            if print_action:
                print("update current action is {}, {}".format(action//self.adg_actions, action%self.adg_actions))
            return True
        else:
            reward[action] = -4000
            return False
            # print("update env success!")

    def init_mask(self):
        self.dfg_mask[:] = 1
        for j in [j for j in range(self.dfg_adj.shape[0]) if np.all(self.dfg_adj[:,j] == 0) and self.dfg_features_vec[j][6] == -1]:
            # print(self.dfg_features_vec)
            # print("j is {}".format(self.dfg_features_vec[j][6]))

            self.dfg_mask[j,:] = 0

        for gib in [id for id, node in  self.adg.getNodes().items() if node.getType() == "GIB"]:
            # print("{} is GIB".format(gib))
            self.mask[:, gib] = 1
            # print(self.mask[:, gib])

        for dfg_id in [node.id for node in self.dfg.getNodes() if node.getOpCode() == "INPUT" or node.getOpCode() == "OUTPUT"]:
            for adg_id in [node.getId() for id,node in self.adg.getNodes().items() if node.getType() == "GPE"]:
                self.mask[dfg_id, adg_id] = 1
        
        for dfg_id in [node.id for node in self.dfg.getNodes() if node.getOpCode() != "INPUT" and node.getOpCode() != "OUTPUT"]:
            for adg_id in [node.getId() for id, node in self.adg.getNodes().items() if node.getType() == "IOB"]:
                self.mask[dfg_id, adg_id] = 1

        print("init mask success")

    def getAct(self, action):
        return action // self.adg_actions, action % self.adg_actions

    def updateFeatures(self, action):
        dfg_act = action // self.adg_actions
        adg_act = action % self.adg_actions
        self.mask[dfg_act,:]=1
        self.mask[:,adg_act]=1
        self.dfg_mask[dfg_act, :] = 1
        print("update features dfg {} adg {}".format(dfg_act, adg_act))

        self.dfg_features_vec[dfg_act][6] = adg_act
        self.adg_features_vec[adg_act][4] = dfg_act

        self.dfg_features_vec[dfg_act][3] =  len(self.dfg_adj[:, dfg_act])
        self.adg_features_vec[adg_act][2] = self.adg.getNodes()[adg_act].getNumInputs()
        self.adg_features_vec[adg_act][3] = self.adg.getNodes()[adg_act].getNumOutputs()
    
    def updateAdj(self, action):
        dfg_act = action // self.adg_actions
        adg_act = action % self.adg_actions
        self.dfg_adj[dfg_act, self.dfg_adj[dfg_act, :]==1] = -1
        self.adg_adj[adg_act,self.adg_adj[adg_act,:]==1] = -1

        for j in [j for j in range(self.dfg_adj.shape[0]) if np.all(self.dfg_adj[:,j] <= 0) and self.dfg_features_vec[j][6] == -1]:
            self.dfg_mask[j,:] = 0
        # self.dfg_features[dg_act].in_degree =

    def isLegalAction(self, action):
        dfg_act = action // self.adg_actions
        adg_act = action % self.adg_actions
        if self.mask[dfg_act, adg_act] == 1:
            return False
        if self.adg.getNode(adg_act).getType() == "GIB":
            return False
        opcode = self.dfg.getNodes()[dfg_act].getOpCode()
        if self.adg.getNode(adg_act).getType() == "IOB" and (opcode != "INPUT" and opcode != "OUTPUT"):
            return False
        if self.adg.getNode(adg_act).getType() == "GPE" and (opcode == "INPUT" or opcode == "OUTPUT"):
            return False
        return True

    def canMapping(self, action, reward):
        dfg_act = action // self.adg_actions
        adg_act = action % self.adg_actions
        # if action in self.reward:
        #     return False
        # reward[action] = -10000
        
        # if not self.isLegalAction(action):
        #     return True
        # print("dfg {} adg {}".format(dfg_act, adg_act))
        # print(self.mask[:, adg_act])
        # print(self.mask[dfg_act, adg_act])
        if self.mask[dfg_act, adg_act] != 0 or self.dfg_mask[dfg_act, adg_act] != 0:
            return False
        
        
        if self.mask[dfg_act, adg_act] == 0 and self.dfg_mask[dfg_act, adg_act] == 0:
            
            #sum_cost = self.adg.getNode(adg_act).getMaxDelay()
            sum_cost = self.get_delay(adg_act)
           
            for src in range(self.dfg_adj.shape[0]):
                if self.dfg_adj[src, dfg_act] == 0:
                    continue
                #print("src {} dfg_act {}".format(src, dfg_act))
                # if self.dfg_adj[src, dfg_act] == -1:
                #     self.dfg_adj[src, dfg_act] = 0
                src_mapped_pe = self.dfg_features_vec[src][6]
                if src_mapped_pe == -1:
                    print("src {} is not mapped".format(src))
                    return False
                dst_mapped_pe = action % self.adg_actions
                #path_cost = self.findPath(src_mapped_pe, dst_mapped_pe)
                #print(".........")
                path_cost = self.cucalateCost(src_mapped_pe, dst_mapped_pe)
                #print("path cost is {}".format(path_cost))
                # if path_cost < 0 :
                #     return False
                sum_cost += path_cost
            # update dfg environment
            # for src in self.dfg_adj[:, action // self.adg_actions]:
            #     # self.dfg_adj[src, action // self.adg_actions] = 0 
            #     self.dfg_features[src].out_degree -= 1
            #     self.dfg_features[src].mapped_pe_id = action % self.adg_actions
            #print("sum cost is {}".format(sum_cost))
            reward[action] = -sum_cost
            return True
        return False 
    
    def findPath(self, src, dst):
        if self.adg_adj[src,dst] == 0:
            return -1
        print("src {} dst {}".format(src,dst))
        min_cost = 100000000
        min_src = ()
        min_dst = ()
        for mid_dst in range(len(self.adg_adj[src,:])):
            # src_node = self.adg.getNodes()[src]
            # dst_node = self.adg.getNodes()[mid_dst]
            if self.adg_adj[src,mid_dst] != 0 and np.any(self.mask[:,mid_dst] == 0):
                # temp = port
                # src_node.useOutPort.add(port[0])
                # dst_node.useInPort.add(port[1])
                self.adg_adj[src,mid_dst] = 0
                mid_delay = self.get_delay(mid_dst)

                if mid_delay == None:
                    mid_delay = 1
                cur_cost = self.findPath(mid_dst, dst) + mid_delay
                if cur_cost < min_cost:
                    if self.adg.getNode(mid_dst).getType() == "GIB":
                        self.adg.adg_features_vec[mid_dst][4] = 1
                        self.adg.adg_features_vec[min_dst][4] = -1
                    min_cost = cur_cost
                    self.adg_adj[src,min_dst] = 1
                    min_dst = mid_dst
                else:
                    self.adg_adj[src,mid_dst] = 1
        if min_cost == 100000000:
            return -1
        return min_cost
    
    def cucalateCost(self, src, dst):
        # if self.adg_adj[src,dst] == 0:
        #     return -1
        # elif self.adg_adj[src,dst] == -1:
        #     self.adg_adj[src,dst] = 0
        #print("src {} dst {}".format(src,dst))
        src_node = self.adg.getNodes()[src]
        dst_node = self.adg.getNodes()[dst]
        # if src_node.getType() == "GIB" or dst_node.getType() == "GIB":
        #     return 0
        src_x = src_node.getX()
        src_y = src_node.getY()
        dst_x = dst_node.getX()
        dst_y = dst_node.getY()
        return abs(src_x - dst_x) + abs(src_y - dst_y)
    
    def get_delay(self, adg_act):
        mid_type = self.adg_features[adg_act].type
        if mid_type == 2:
            return 1
        elif mid_type == 0:
            return 2
        else:
            return 3


    
    
    def updateMeta(self, action):
        ## UCT choose next node
        dfg_act = action // len(self.adg_features)
        self.meta = self.adg_features[dfg_act]

    def reset(self):
        self.mask = np.zeros((len(self.dfg.getNodes()), self.adg.getMaxNodeId()+1))
        self.dfgDeatures = DFGFeatures(self.init_env[0])
        self.dfg_features = self.dfgDeatures.features

        self.dfg_features_vec = self.dfgDeatures.getFeaturesVector(self.init_env[2])
        self.adgDeatures = ADGFeatures(self.init_env[1])
        self.adg_features = self.adgDeatures.features
        self.adg_features_vec = self.adgDeatures.getFeaturesVector()
        self.dfg_adj = getDfgAdj(self.init_env[0])
        self.adg_adj = getAdgAdj(self.init_env[1])
        self.init_mask()

    def draw(self):
        """draw the environment"""
        self.graph_visual.draw()


"""TODO: refactor the mapping state and Environment"""  
class MappingState:
    def __init__(self, env, action = None):
        super().__init__()
        self.last_action = action
        self.env = copy.deepcopy(env)
        # self.dfg_mask = {}
        # self.adg_mask = {}
        self.mask = self.env.mask #np.zeros((len(env.dfg.getNodes()), env.adg_actions))
        
        # self.dfg_features = DFGFeatures(env.dfg)
        # self.dfg_features_vec = self.dfg_features.getFeaturesVector(env.ops)
        # self.adg_features = ADGFeatures(env.adg)
        # self.adg_features_vec = self.adg_features.getFeaturesVector()
        # self.init_mask()
        # self.dfg_adj = getDfgAdj(env.dfg)
        # self.adg_adj = getAdgAdj(env.adg)
        self.reward = {}
        # self.action_map = {}
        self.sum_reward = 0
        self.mapping_state = True
        # self.plot = plt.subplots()
    
    def clone(self): 
        new_state = copy.deepcopy(self)
        new_state.reward = {}
        return new_state
    
    def init_mask(self):
        for gib in [node.getId() for node in  self.env.adg.getNodes if node.getType() == "GIB"]:
            self.mask[:, gib] = 1

        for dfg_id in [node.getId() for node in self.env.dfg.getNodes() if node.getOpCode() == "INPUT" or node.getOpCode() == "OUTPUT"]:
            for adg_id in [node.getId() for node in self.env.adg.getNodes() if node.getType() == "GPE"]:
                self.mask[dfg_id, adg_id] = 1
        
        for dfg_id in [node.getId() for node in self.env.dfg.getNodes() if node.getOpCode() != "INPUT" and node.getOpCode() != "OUTPUT"]:
            for adg_id in [node.getId() for node in self.env.adg.getNodes() if node.getType() == "IOB"]:
                self.mask[dfg_id, adg_id] = 1
    
    def is_terminal(self):
        return np.all(self.mask > 0) or len(self.get_actions()) == 0 or self.mapping_state == False
    
    def get_actions(self):
        actions = []
        for i in range(len(self.env.dfg_features_vec)):  
            # indegree, prececessors = (self.env.dfgFeatures).getInDegreeAndPredecessors(i)
            # has_unmapping_prececessor = False
            # for processor in prececessors:
            #     if self.mask[processor,:].sum() < self.env.adg_actions:
            #         has_unmapping_prececessor = True
            # if has_unmapping_prececessor:
            #     continue
            if np.all(self.env.dfg_mask[i,:] == 1):
                continue

            # print("legal dfg action is {}".format(i))
            # print(self.env.dfg_adj[:,i])
            # print(np.where(self.mask[i,:] == 0))
            if np.any(self.mask[i,:] == 0):
                for j in range(self.env.adg_actions):
                    if np.any(self.mask[:,j] == 0):
                        if self.env.adg.getNode(j) != None and self.env.mask[i,j] == 0:
                            action = i * self.env.adg_actions + j
                            actions.append(action)
            # print(actions)
        return actions
    
    def get_distance(self, cur_action):
        if self.last_action == None:
            return 1
        cur_adg = cur_action % self.env.adg_actions
        last_adg = self.last_action % self.env.adg_actions
        return self.env.cucalateCost(last_adg, cur_adg)

    
    def take_action(self, action, print_action = False):
        self.last_action = action
        if print_action:
            print("current action is {}, {}".format(action//self.env.adg_actions, action%self.env.adg_actions))
        if not self.env.update(action, self.reward, print_action):
            self.mapping_state = False
            self.sum_reward *= -10
        self.sum_reward += self.reward[action] # add reward to sum_reward
        # dfg_act = action // self.env.adg_actions
        # adg_act = action % self.env.adg_actions
        # self.mask[dfg_act,:] = 1
        # self.mask[:,adg_act] = 1
        
    
    def get_reward(self):
        # self.
        return self.sum_reward
    
    def take_ramdom_action(self):
        actions = self.get_actions()
        action = random.choice(actions)
        self.take_action(action)

    
    def get_policy_action(self, policy):
        action = policy.argmax()
        return action
    
    def draw(self):
        """draw the visual state"""

        self.env.graph_visual.draw_state(self)
    
class GraphEmbedding(nn.Module):
    def __init__(self, dfg_shape, adg_shape, device):
        super().__init__()
        self.device = device
        # network components
        # self.fc = nn.Linear(,1)
        self.fc96 = nn.Linear(64,32)
        self.fc32 = nn.Linear(32,32)
        self.leakyrelu = nn.LeakyReLU(0.2)

        # self.dfg_reduce_fc = nn.Linear(dfg_shape, 32)
        # self.adg_reduce_fc = nn.Linear(adg_shape, 32)

        self.dfg_gat = GAT(dfg_shape, 8, 32, 0.6, 0.2, 8)
        self.adg_gat = GAT(adg_shape, 8, 32, 0.6, 0.2, 8)
    
    def forward(self, batch_env):
        # dfg embedding
        batch_dfg_features_vec = []
        batch_adg_features_vec = []
        batch_dfg_adj = []
        batch_adg_adj = []
        for e in batch_env:
            batch_dfg_features_vec.append(e.dfg_features_vec)
            batch_adg_features_vec.append(e.adg_features_vec)
            e.dfg_adj[e.dfg_adj == -1] = 0
            batch_dfg_adj.append(e.dfg_adj)
            e.adg_adj[e.adg_adj == -1] = 0
            batch_adg_adj.append(e.adg_adj)

        
        batch_dfg_features_vec = np.array(batch_dfg_features_vec)
        batch_adg_features_vec = np.array(batch_adg_features_vec)
        batch_dfg_adj = np.array(batch_dfg_adj)
        batch_adg_adj = np.array(batch_adg_adj)

        dfg_x = self.dfg_gat(torch.FloatTensor(batch_dfg_features_vec).to(self.device), torch.FloatTensor(batch_dfg_adj).to(self.device))

        dfg_x = torch.mean(dfg_x, dim=1, keepdim=True)
        #dfg_x = self.dfg_reduce_fc(dfg_x)


        # adg embedding
        adg_x = self.adg_gat(torch.FloatTensor(batch_adg_features_vec).to(self.device), torch.FloatTensor(batch_adg_adj).to(self.device))
        adg_x = torch.mean(adg_x, dim=1, keepdim=True)

        # current mapped node information
        # meta = self.fc(env.dfg_features[env.last_action])
        # adg_x = self.adg_reduce_fc(adg_x)
        x = torch.cat([dfg_x, adg_x], dim=2)

        # mlp fc 32-32
        x = F.relu(self.fc96(x))
        x = F.relu(self.fc32(x))

        # output
        x = self.leakyrelu(x)

        return x

"""set leraning rate for optimizer"""
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.fc = nn.Linear(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, mask):
        x = F.relu(self.fc(x))
        x = self.leakyrelu(x)
        
        #x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        # x = F.normalize(x, p=1, dim=1)
        
        x = F.log_softmax(x, dim=2)
       
        #print("x shape {}".format(x[...,1:100]))
        x = x * mask
        # print(x[...,1:100])
        # x = x.reshape(-1)
        return x

class ValueNet(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(32, 1)
        self.mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.mlp(x)
        x = self.leakyrelu(x)
        x = self.fc(x)
        return x
    
class PolicyValueNet(nn.Module):
    def __init__(self,dfg_shape, adg_shape, state_size, action_size, hidden_size=32, device="cpu"):
        super().__init__()
        self.embedding = GraphEmbedding(dfg_shape, adg_shape, device)
        self.policyNet = PolicyNet(state_size, action_size, hidden_size)
        self.valueNet = ValueNet()

    def forward(self, state, mask):
        x = self.embedding(state)
        log_act_probs = self.policyNet(x, mask)
        value = self.valueNet(x)
        return log_act_probs, value




class AgentNetwork():
    def __init__(self, dfg_shape, adg_shape, state_size, action_size, hidden_size=32, use_gpu=False, model_file=None):
        # self.graph_embedding = GraphEmbedding()
        super().__init__()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")

        self.agentNet = PolicyValueNet(dfg_shape, adg_shape, state_size, action_size, hidden_size, device=self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.agentNet.parameters(), lr=0.01)

        if model_file:
            self.agentNet = torch.load(model_file, map_location=self.device)# load_state_dict(net_params)

    def policy_value(self, batch_state):
        # ! state is a  batch list of states
        # ! fix this for batch processing
        # x = self.embedding(state_embedding)
        batch_env = [state.env for state in batch_state]
        batch_mask = [state.env.mask.flatten() for state in batch_state]
        log_act_probs, value = self.agentNet(batch_env, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device))
        act_probs = np.exp(log_act_probs.cpu().data.numpy())

        value = value.cpu().data.numpy()

        return act_probs, value
    
    def policy_value_pure(self, batch_state):
        batch_env = [state.env for state in batch_state]
        batch_mask = np.array([state.mask.flatten() for state in batch_state])
        log_act_probs, value = self.agentNet(batch_env, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device))
        return log_act_probs, value
    
    def policy_value_fn(self, state):
        legal_actions = state.get_actions()
        mask = state.mask * state.env.dfg_mask
        log_act_probs, value= self.agentNet([state.env],torch.FloatTensor(mask.flatten()).to(self.device))
        log_act_probs = log_act_probs.reshape(-1)
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()[0][0]
        act_probs = zip(legal_actions, act_probs[legal_actions])
        return act_probs, value

    
    def train_step(self, state_batch,  mcts_probs, sum_latency, lr):
        """ perform a training step"""
        if self.use_gpu:
            state_batch = state_batch
            mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
            sum_latency = torch.FloatTensor(sum_latency).to(self.device)
        else:
            state_batch = state_batch
            mcts_probs = mcts_probs
            sum_latency = sum_latency
        # reset gradients
        self.optimizer.zero_grad()

        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        act_probs, value = self.policy_value_pure(state_batch)
        # value = torch.FloatTensor(value).to(self.device)
        # act_probs = torch.FloatTensor(act_probs).to(self.device)
        

        value_loss = F.mse_loss(value.view(-1), sum_latency)
        
        policy_loss = -torch.mean(torch.sum(mcts_probs * act_probs, 2))

        # backward
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        

        # calc policy entropy, for monitoring only
       
        entropy = -torch.mean(torch.sum(act_probs * torch.exp(act_probs), 2))
        return loss.item(), entropy.item()
    

# class GATAutoEncoder