from model import *

from dfgParser import DFGParser
from loadData import getDfgAdj

from dfg import *
from adgParser import ADGIR
from mcts import MCTSPlayer
from mapping import Mapping

import torch
from tqdm import tqdm
import pickle
# from mcts import ma
import matplotlib.pyplot as plt

from operations import Operation, Operations   
parser = DFGParser('dfg.json')
adg_parser = ADGIR('cgra_adg.json')
operations = Operations()
operations.OpParser('operations.json')
name2op = operations.name2op
dfg = parser.getDFG()
adg = adg_parser.getADG()
# if torch.cuda.is_available():
#     print("CUDA is available. You can use GPU.")
# else:
#     print("CUDA is not available. Please check your CUDA installation.")

model = GAT(nfeat=7, nhid=8, nclass=7, dropout=0.6, alpha=0.2, nheads=8)
features = DFGFeatures(dfg).getFeaturesVector(operations)
adg_features = ADGFeatures(adg).getFeaturesVector()
# print(features)
# print(adg_features)
# adj = getDfgAdj(dfg)

# out = model(torch.tensor(features, dtype=torch.float32), torch.tensor(adj, dtype=torch.float32))

# print(out)


class MCTS_RL:
    # init
    def __init__(self, dfg, adg, ops):

        """some parameters for training"""
        self.batch_num = 20
        self.kl_targ = 0.02
        self.check_freq = 1
        self.target_latency = 100
        self.best_latency = 1000
        self.lr_multiplier = 1.0
        self.learn_rate = 2e-3
        self.epoches = 1
        self.use_gpu = torch.cuda.is_available()
        self.state_size = 32 
        #self.action_size = 400

        self.dfg_adj = getDfgAdj(dfg)
        dfgFeatures = DFGFeatures(dfg)
        self.dfg_features = dfgFeatures.getFeaturesVector(ops)
        self.adg_adj = getAdgAdj(adg)
        self.adg_features = ADGFeatures(adg).getFeaturesVector()
        # self.mask = np.zeros((len(dfg.getNodes())+4, len(adg.getNodes())+30))
        self.action_size = len(dfg.getNodes()) * (adg.getMaxNodeId()+1)
        self.buffer = []
        
        self.env = Environment(dfg, adg, ops)
        # self.embedding = GraphEmbedding(7, 7)
        actions_len = len(dfg.getNodes()) * (adg.getMaxNodeId()+1)
        print(actions_len)
        self.agent_net = AgentNetwork(7, 5, 32, actions_len,32, self.use_gpu, model_file="models/best-agent.pt")
        self.mapping = Mapping(dfg, adg, ops)
        self.mcst = MCTS(self.agent_net)
        self.agent = MCTSPlayer(self.agent_net)

    def collect_data(self, n_iters):
        for iter in range(n_iters):
            # print("current iter {}".format(iter))
            latency, data = self.mapping.start_mapping(self.agent)
            print("collect data... | current latency: {}".format(latency))
            data = list(data)[:]
            # data = self.get_equi_data(data)
            self.buffer.extend(data)


    def get_equi_data(self, data):
        equi_data = []
        for state, prob, reward in data:
            for i in [1,2,3,4]:
                equi_data.append((self.get_equi_state(state, i), prob, reward))
        return equi_data
    
    def policy_update(self):
        """update the policy-value network"""
        mini_batch = random.sample(self.buffer, 15)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        

        # ! state batch need to GAT embedding!
        mask_batch = np.zeros((len(state_batch), self.action_size))
        state_batches = np.zeros((len(state_batch), 32))
        #enves = [state.env for state in state_batch]
        for i in range(len(state_batch)):
            mask_batch[i] = state_batch[i].mask.reshape(-1)
       
        # state_batches = self.agent_net.embedding(state_batch[0].env)
        
        old_probs, old_v = self.agent_net.policy_value(state_batch)
        old_probs = np.exp(old_probs)
        old_v = old_v

        for i in range(self.epoches):
            loss, entropy = self.agent_net.train_step(state_batch, mcts_probs_batch, reward_batch, self.learn_rate*self.lr_multiplier)

            new_probs, new_v = self.agent_net.policy_value(state_batch)
            new_probs = np.exp(new_probs)
            new_v = new_v

            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        print(reward_batch)
        print("-----------------")
        # print(old_v.flatten())

        explained_var_old = 1 - np.var(np.array(reward_batch) - old_v.flatten()) / np.var(np.array(reward_batch))
        explained_var_new = 1 - np.var(np.array(reward_batch) - new_v.flatten()) / np.var(np.array(reward_batch))

        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}").format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy
    
    def policy_evaluate(self, n_epoches=1):
        current_mcts_agent = MCTSPlayer(self.agent_net)
        sum_latency = 0
        for i in range(n_epoches):
            latency, data = self.mapping.start_mapping(current_mcts_agent)
            sum_latency -= latency
        return sum_latency / n_epoches



    def run(self, batch_num):
        """ a train pipeline for mapping """
        try:
            losses = []
            for i in tqdm(range(self.batch_num)):
                self.collect_data(50)
                if len(self.buffer) > 1:
                    loss, entropy = self.policy_update()
                    losses.append(loss)
                if (i+1)%self.check_freq == 0:
                    print("current batch: {}".format(i+1))
                    torch.save(self.agent_net.agentNet, './models/cur-agent.pt')
                    # TODO: fix the policy_value function
                    cur_latency = self.policy_evaluate()
                    print("current latency: {}".format(cur_latency))
                    if cur_latency < self.best_latency:
                        self.best_latency = cur_latency
                        torch.save(self.agent_net.agentNet, './models/best-agent.pt')

            plt.plot(losses)
            plt.ylabel('loss')
            plt.savefig("./figures/loss.png")
            with open("./models/mcts.pkl", "wb") as f:
                pickle.dump(self.mcst, f)

        except KeyboardInterrupt:
            print('\n\rquit')
        


rl = MCTS_RL(dfg, adg, operations)
rl.run(10)


    


