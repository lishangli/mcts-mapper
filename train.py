# from model import *
from mcts import MCTS, MCTSPlayer
from net import AgentNetwork
from GAT import GAT
from state import Environment


from dfgParser import DFGParser
from loadData import getDfgAdj

from dfg import *
from adg import ADGFeatures
from loadData import getAdgAdj
from adgParser import ADGIR
from mcts import MCTSPlayer
from mapping import Mapping
from dataLoader import JSONDataset

import torch
from tqdm import tqdm
from rich.progress import track, Progress
import pickle
import random

# from mcts import ma
import matplotlib.pyplot as plt

from operations import Operation, Operations

parser = DFGParser("dfg.json")
adg_parser = ADGIR("cgra_adg.json")
operations = Operations()
operations.OpParser("operations.json")
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
    def __init__(self, adg, agent_net):
        """some parameters for training"""
        self.batch_num = 10
        self.batch_size = 50
        self.kl_targ = 0.02
        self.check_freq = 10
        self.target_latency = 100
        self.best_latency = 1000
        self.lr_multiplier = 1.0
        self.learn_rate = 2e-3
        self.epoches = 100
        self.use_gpu = torch.cuda.is_available()
        self.state_size = 32
        # self.action_size = 400

        # self.dfg_adj = getDfgAdj(dfg)
        # dfgFeatures = DFGFeatures(dfg)
        # self.dfg_features = dfgFeatures.getFeaturesVector(ops)
        self.adg_adj = getAdgAdj(adg)
        self.adg_features = ADGFeatures(adg).getFeaturesVector()
        # self.mask = np.zeros((len(dfg.getNodes())+4, len(adg.getNodes())+30))
        self.action_size = adg.getNodeNums() * (adg.getMaxNodeId() + 1)
        print("adg node nums: {}".format(adg.getNodeNums()))
        print("adg max node id: {}".format(adg.getMaxNodeId()))
        self.buffer = []
        self.route_buffer = []

        # self.env = Environment(dfg, adg, ops)

        # self.embedding = GraphEmbedding(7, 7)
        actions_len = self.action_size
        route_size = len(adg.getEdges())
        self.route_actions_size = len(adg.getEdges())
        self.agent_net = agent_net
        # AgentNetwork(
        #     7,
        #     5,
        #     32,
        #     actions_len,
        #     route_size,
        #     32,
        #     self.use_gpu,
        #     model_file=None,
        # )
        self.agent = MCTSPlayer(self.agent_net)
        # self.mapping = Mapping(dfg, adg, ops)
        self.mcst = MCTS(self.agent_net)

    def collect_data(self, n_iters, dfgs, progress, task):
        # self.agent.step()
        print("agent step {}".format(self.agent.t))
        for iter in range(n_iters):
            for dfg in dfgs:
                progress.update(task, advance=1)
                # self.mapping.use_true_reward()
                mapping_task = Mapping(dfg, adg, operations)
                latency, data, route_data = mapping_task.start_mapping(self.agent)
                print("collect data... | current latency: {}".format(latency))
                data = list(data)[:]
                route_data = list(route_data)[:]
                # data = self.get_equi_data(data)
                self.buffer.extend(data)
                self.route_buffer.extend(route_data)

    def clean_data(self):
        self.buffer = []
        self.route_buffer = []

    def collect_ppo_data(self, n_iters, progress, task):
        for iter in range(n_iters):
            progress.update(task, advance=1)
            # self.mapping.use_true_reward()
            latency, data, route_data = self.mapping.ppo_mapping(self.agent)
            print("collect data... | current latency: {}".format(latency))
            data = list(data)[:]
            # data = self.get_equi_data(data)
            self.buffer.extend(data)
            self.route_buffer.extend(route_data)

    def get_equi_data(self, data):
        equi_data = []
        for state, prob, reward in data:
            for i in [1, 2, 3, 4]:
                equi_data.append((self.get_equi_state(state, i), prob, reward))
        return equi_data

    def masked_kl_divergence(self, probs_old, probs_new, mask):
        valid_actions = mask.astype(np.float32)
        # print("old probs shape is {}, new probs shape is {}".format(log_probs_old.shape, log_probs_new.shape))
        kl = (
            probs_old
            * (np.log(probs_old + 1e-10) - np.log(probs_new + 1e-10))
            * valid_actions
        )
        # print("kl shape is {}. mask shape is {}".format(kl.shape, mask.shape))
        kl_sum = kl.sum(axis=1)
        mask_sum = valid_actions.sum(axis=1) + 1e-8
        # print("kl sum is {}".format(kl_sum))
        # print("kl sum shape is {}. mask sum shape is {}".format(kl_sum.shape, mask_sum.shape))
        return (kl_sum / mask_sum).mean()

    def policy_update(self):
        """update the policy-value network using"""
        # print("buffer data: {}".format(self.buffer))
        mini_batch = random.sample(self.buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        value_batch = [data[2] for data in mini_batch]
        reward_batch = [data[3] for data in mini_batch]
        action_batch = [data[4] for data in mini_batch]

        policy_value_fn = self.agent_net.policy_value
        reward_model = self.agent_net.reward_model
        # mask_batch = np.zeros((len(state_batch), self.action_size))
        # for i in range(len(state_batch)):
        #     mask_batch[i] = state_batch[i].mask.reshape(-1)

        mask_batch = np.array([s.mask.reshape(-1) for s in state_batch])

        old_probs, old_v = policy_value_fn(state_batch)
        old_reward = reward_model(state_batch, action_batch)

        # ! state batch need to GAT embedding!
        # mask_batch = np.zeros((len(state_batch), self.action_size))
        # state_batches = np.zeros((len(state_batch), 32))
        # enves = [state.env for state in state_batch]
        # for i in range(len(state_batch)):
        #     mask_batch[i] = state_batch[i].mask.reshape(-1)

        # state_batches = self.agent_net.embedding(state_batch[0].env)

        # old_probs, old_v = policy_value_fn(state_batch)
        # old_reward = reward_model(state_batch, action_batch)
        # old_probs = np.exp(old_probs)
        # old_v = old_v

        for i in range(self.epoches):

            loss, entropy = self.agent_net.train_step_pretrain(
                state_batch,
                mcts_probs_batch,
                value_batch,
                reward_batch,
                action_batch,
                self.learn_rate * self.lr_multiplier,
            )

            new_probs, new_v = policy_value_fn(state_batch)
            new_reward = reward_model(state_batch, action_batch)
            # new_probs = np.exp(new_probs)
            # new_v = new_v

            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            # kl = self.masked_kl_divergence(old_probs, new_probs, mask_batch)

            print(
                "old_probs: {},\n new_probs: {}, kl: {}".format(
                    old_probs, new_probs, kl
                )
            )
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten())
        # valid_values = (mask_batch == 1)* value_batch
        explained_var_old = 1 - np.var(
            np.array(value_batch) - old_v.flatten()
        ) / np.var(np.array(value_batch))
        explained_var_new = 1 - np.var(
            np.array(value_batch) - new_v.flatten()
        ) / np.var(np.array(value_batch))
        # explained_var_old = 1 - np.var(valid_values - old_v.flatten()) / np.var(valid_values)
        # explained_var_new = 1 - np.var(valid_values - new_v.flatten()) / np.var(valid_values)

        print(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def route_policy_update(self):
        mini_batch = random.sample(self.route_buffer, 50)
        state_batch = [data[0] for data in mini_batch]
        print("state_batch: {}".format(state_batch))
        mcts_probs_batch = [data[1] for data in mini_batch]
        value_batch = [data[2] for data in mini_batch]

        policy_value_fn = self.agent_net.route_policy_value
        mask_batch = np.zeros((len(state_batch), self.route_actions_size))
        for i in range(len(state_batch)):
            mask_batch[i] = state_batch[i].route_mask.reshape(-1)

        old_probs, old_v = policy_value_fn(state_batch)
        # old_probs = np.exp(old_probs)
        old_v = old_v

        for i in range(self.epoches):
            loss, entropy = self.agent_net.train_route_step(
                state_batch,
                mcts_probs_batch,
                value_batch,
                self.learn_rate * self.lr_multiplier,
            )

            new_probs, new_v = policy_value_fn(state_batch)
            # new_probs = np.exp(new_probs)
            new_v = new_v

            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )

            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten())

        explained_var_old = 1 - np.var(
            np.array(value_batch) - old_v.flatten()
        ) / np.var(np.array(value_batch))
        explained_var_new = 1 - np.var(
            np.array(value_batch) - new_v.flatten()
        ) / np.var(np.array(value_batch))

        print(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def policy_evaluate(self, test_dfg, n_epoches=1):
        current_mcts_agent = MCTSPlayer(self.agent_net, 10)
        sum_latency = 0
        for i in range(n_epoches):
            # self.mapping.use_true_reward()
            mapping_task = Mapping(test_dfg, adg, operations)
            latency, data, route_data = mapping_task.start_mapping(current_mcts_agent)
            sum_latency -= latency
        return sum_latency / n_epoches

    def run(self, batch_num, dfgs):
        """a train pipeline for mapping"""
        try:
            losses = []
            test_dfg = DFGParser("dfg.json").getDFG()
            with Progress() as progress:
                train_task = progress.add_task("training...", total=batch_num)
                collect_task = progress.add_task("collecting data...", total=4)
                for i in range(self.batch_num):
                    progress.update(train_task, advance=1)
                    self.collect_data(1, dfgs, progress, collect_task)
                    # if i > self.batch_num * 0.2:
                    #     self.mapping.use_true_reward()
                    if len(self.buffer) >= self.batch_size:
                        loss, entropy = self.policy_update()
                        self.route_policy_update()
                        losses.append(loss)
                        self.agent.step()
                    if (i + 1) % self.check_freq == 0:
                        print("current batch: {}".format(i + 1))
                        torch.save(self.agent_net.agentNet, "./models/cur-agent.pt")
                        # TODO: fix the policy_value function
                        cur_latency = self.policy_evaluate(test_dfg)
                        print("current latency: {}".format(cur_latency))
                        if cur_latency < self.best_latency:
                            self.best_latency = cur_latency
                            torch.save(
                                self.agent_net.agentNet, "./models/best-agent.pt"
                            )

                plt.plot(losses)
                plt.ylabel("loss")
                plt.savefig("./figures/loss.png")
                with open("./models/mcts.pkl", "wb") as f:
                    pickle.dump(self.mcst, f)

        except KeyboardInterrupt:
            print("\n\rquit")


actions_len = adg.getNodeNums() * (adg.getMaxNodeId() + 1)
route_size = len(adg.getEdges())

route_size = len(adg.getEdges())
agent_net = AgentNetwork(
    7,
    5,
    32,
    actions_len,
    route_size,
    32,
    True,
    model_file=None,
)
data_loader = JSONDataset("./dataset/microbench/json", operations, transform=None)
rl = MCTS_RL(adg, agent_net)
dfgs = []
for data in data_loader:
    dfg = data["dfg"]
    dfgs.append(dfg)
rl.run(2, dfgs)

# dfgs = []
# dfgs.append(dfg)
# rl.run(2, dfgs)
# rl.clean_data()
