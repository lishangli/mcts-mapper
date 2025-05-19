# from model import *
from parser import DFGParser, ADGFeatures
from agent import MCTS, MCTSPlayer
from models import AgentNetwork

from utils import getAdgAdj
from mapper import Mapping
from env import MappingState
from parser import ADGIR, Operations, ADGFeatures

import torch
import numpy as np
from dataclasses import dataclass
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from rich.progress import Progress
import pickle
import random
import multiprocessing as mp
import concurrent.futures
from functools import partial
from typing import List, Tuple, Union
import csv
import os
import matplotlib.pyplot as plt

# GOLBAL SETTING

adg_parser = ADGIR("../example/cgra_adg.json")
GLOBAL_ADG = adg_parser.getADG()
GLOBAL_OPERATIONS = Operations()
GLOBAL_OPERATIONS.OpParser("../example/operations.json")

stop_event = mp.Event()

class ProcessData:
    def __init__(self,adg, dfg, ops):
        self.adg = adg
        self.dfg = dfg
        self.ops = ops


def collect_data_multi(self, n_iters, dfgs, num_workers=4):
    """
    Parallel version of data collection using multiprocessing.

    Args:
        n_iters: Number of iterations
        dfgs: List of DFGs to process
        progress: Progress bar object
        task: Task object for progress tracking
        num_workers: Number of parallel workers to use
    """

    results = []
    # global GLOBAL_ADG, GLOBAL_OPERATIONS
    # print("before change global varible")
    # GLOBAL_ADG = self.adg
    # GLOBAL_OPERATIONS = self.operations
    # print("after change global varible")

    # task = progress.add_task("collecting data...", total=len(dfgs))
    # Create a shared agent for all processes
    # Note: You might need to implement agent sharing or copying mechanism
    # depending on your actual agent implementation
    state_dict = self.agent_net.agentNet.state_dict()  # This might need customization

    agent_config = self.agent_net.agent_config
    # 定义子进程初始化函数
    for iter in range(n_iters):
        # Set up multiprocessing resources

        with mp.get_context("spawn").Pool(
            processes=num_workers,
            ) as pool:
            # Create partial function with fixed arguments
            process_func = partial(
                process_dfg,
                agent_net_state_dict=state_dict,
                agent_config=agent_config,
                adg_path = self.adg_path,
                operations_path = self.operations_path
            )
            result = pool.map(process_func, dfgs)
            results.extend(result)

        stop_event.set()
    ave_r = 0
    for r, data, route_data in results:
        self.buffer.extend(data)
        self.route_buffer.extend(route_data)
        ave_r += r 

    self.latencies.append(ave_r)


def process_dfg(dfg, agent_net_state_dict, agent_config, adg_path, operations_path):
    """Process a single DFG and return the collected data"""
    # global GLOBAL_ADG, GLOBAL_OPERATIONS
    try:
        adg_parser = ADGIR(adg_path)
    
        operations = Operations()
        operations.OpParser(operations_path)
        
        adg = adg_parser.getADG()
        mapping_task = Mapping(dfg, adg, operations)
        agent_net = AgentNetwork(**agent_config)

        agent_net.agentNet.load_state_dict(agent_net_state_dict)
        agent = MCTSPlayer(agent_net)
        latency, data, route_data = mapping_task.grpo_mappingv2(agent)
        tqdm.write(f"collect data... | current latency: {latency}")

        # # progress.update(task, advance=1)
        agent_net.agentNet.to("cpu")  # 将模型移动到 CPU
        del agent_net  # 删除模型对象
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return latency, data, route_data

        # return []
    except Exception as e:
        print(f"Error processing DFG: {e}")
        return None, [], []


class MCTS_RL:
    # init
    def __init__(self, adg, operations, agent_net):
        """some parameters for training"""
        self.batch_num = 800
        self.batch_size = 64
        self.kl_targ = 0.1
        self.route_kl_targ = 0.1                                                   
        self.check_freq = 5
        self.target_latency = 100
        self.best_latency = 1000
        self.lr_multiplier = 1.0
        self.route_lr_multiplier = 1.0
        self.learn_rate = 3e-4
        self.epoches = 50
        self.use_gpu = torch.cuda.is_available()
        self.state_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adg_path = adg.path
        self.operations_path = operations.path
        self.adg_adj = getAdgAdj(adg)
        self.adg_features = ADGFeatures(adg).getFeaturesVector()
        # self.mask = np.zeros((len(dfg.getNodes())+4, len(adg.getNodes())+30))
        self.action_size = adg.getNodeNums() * (adg.getMaxNodeId() + 1)
        self.buffer = []
        self.route_buffer = []
        self.route_actions_size = len(adg.getEdges())
        self.agent_net = agent_net
        # )
        self.agent = MCTSPlayer(self.agent_net)
        # self.mapping = Mapping(dfg, adg, ops)
        self.mcst = MCTS(self.agent_net)

        # some variables for figures saving
        self.kls = []
        self.explained_vars = []

        self.route_kls = []
        self.route_explained_vars = []

        self.losses = []
        self.pref = []
        self.latencies = []

        self.value_loss = []
        self.policy_loss = []

        self.route_value_loss = []
        self.route_policy_loss = []

    @staticmethod
    def _prepare_batch_data(
        batch_state: List[MappingState], device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Prepare batch data for processing"""
        batch_data = {
            "dfg_features": [],
            "adg_features": [],
            "dfg_adj": [],
            "adg_adj": [],
            "dfg_meta": [],
            "adg_meta": [],
        }

        max_dfg_nodes = 30

        for state in batch_state:
            # Clean adjacency matrices
            dfg_adj = state.env.dfg_adj.copy()
            adg_adj = state.env.adg_adj.copy()
            dfg_adj[dfg_adj == -1] = 0
            adg_adj[adg_adj == -1] = 0

            # 2 dfg_features_vec padding
            dfg_features_vec = state.env.dfg_features_vec
            num_dfg_nodes = dfg_features_vec.shape[0]

            dfg_features_vec_padded = dfg_features_vec

            if num_dfg_nodes < max_dfg_nodes:
                dfg_features_vec_padded = np.pad(
                    dfg_features_vec,
                    ((0, max_dfg_nodes - num_dfg_nodes), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            dfg_adj_padded = dfg_adj
            num_dfg_nodes_adj = dfg_adj_padded.shape[0]
            if num_dfg_nodes_adj < max_dfg_nodes:
                padding_rows_adj = max_dfg_nodes - num_dfg_nodes_adj
                dfg_adj_padded = np.pad(
                    dfg_adj,
                    ((0, padding_rows_adj), (0, padding_rows_adj)),
                    mode="constant",
                    constant_values=0,
                )  # 使用 np.pad 进行填充

            dfg_meta = state.get_dfg_meta()

            dfg_meta_padded = dfg_meta
            num_dfg_nodes_meta = dfg_meta.shape[0]
            if num_dfg_nodes_meta < max_dfg_nodes:
                dfg_meta_padded = np.pad(
                    dfg_meta,
                    ((0, max_dfg_nodes - num_dfg_nodes_meta), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # Collect batch data
            batch_data["dfg_features"].append(dfg_features_vec_padded)
            batch_data["adg_features"].append(state.env.adg_features_vec)
            batch_data["dfg_adj"].append(dfg_adj_padded)
            batch_data["adg_adj"].append(adg_adj)
            batch_data["dfg_meta"].append(dfg_meta_padded)
            batch_data["adg_meta"].append(state.get_adg_meta())

        # Convert to numpy arrays
        return tuple(
            torch.FloatTensor(np.array(v)).to(device) for v in batch_data.values()
        )

    def collect_data(self, n_iters, dfgs, progress, task):
        # self.agent.step()
        # print("agent step {}".format(self.agent.t))
        for iter in range(n_iters):
            for dfg in dfgs:
                progress.update(task, advance=1)
                # self.mapping.use_true_reward()
                mapping_task = Mapping(dfg, self.adg, self.operations)
                latency, data, route_data = mapping_task.start_mapping(self.agent)
                tqdm.write(f"collect data... | current latency: {latency}")
                # data = list(data)[:
                self.buffer.extend(data)
                self.route_buffer.extend(route_data)

    def collect_data_multithread(self, n_iters, dfgs, max_workers=4):
        all_data = []
        all_route_data = []

        def process_single_dfg(dfg, agent_net_state_dict, agent_config, progress, task):
            """Process a single DFG and return the collected data"""
            try:

                print("enter process_dfg")
                # collect_task = progress.add_task("collect task")
                mapping_task = Mapping(dfg, self.adg, self.operations)
                agent_net = AgentNetwork(**agent_config)

                agent_net.agentNet.load_state_dict(agent_net_state_dict)
                agent = MCTSPlayer(agent_net)
                latency, data, route_data = mapping_task.start_mapping(agent)
                tqdm.write(f"collect data... | current latency: {latency}")
                progress.update(task, advance=1)
                return data, route_data
                # return []
            except Exception as e:
                print(f"Error processing DFG: {e}")
                return [], []

        # Use ThreadPoolExecutor for CPU-bound tasks
        # ProcessPoolExecutor could be used instead if the agent can be pickled
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        state_dict = (
            self.agent_net.agentNet.state_dict()
        )  # This might need customization
        agent_config = self.agent_net.agent_config
        with Progress() as progress:

            for iter in range(n_iters):
                task = progress.add_task("collecting data...", total=len(dfgs))
                # Create tasks for all DFGs in the current iteration
                futures = []
                for dfg in dfgs:
                    # Submit task to executor
                    process_func = partial(
                        process_single_dfg,
                        agent_net_state_dict=state_dict,
                        agent_config=agent_config,
                        progress=progress,
                        task=task,
                    )

                    future = executor.submit(process_func, dfg)
                    futures.append(future)

                # Process completed futures as they finish
                for future in concurrent.futures.as_completed(futures):
                    try:
                        data, route_data = future.result()
                        all_data.extend(data)
                        all_route_data.extend(route_data)
                    except Exception as e:
                        print(f"Error getting result: {e}")

        # Ensure executor is shut down properly
        executor.shutdown()

        # Update buffers with all collected data
        self.buffer.extend(all_data)
        self.route_buffer.extend(all_route_data)

    def clean_data(self):
        self.buffer = []
        self.route_buffer = []

    def collect_ppo_data(self, n_iters,dfgs, progress, task):
        for dfg in dfgs:
            for iter in range(n_iters):
                progress.update(task, advance=1)
                mapping_task = Mapping(dfg, self.adg, self.operations)
                latency, data, route_data = mapping_task.ppo_mapping(self.agent)
                tqdm.write(f"collect data... | current latency: {latency}")
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
    
    def get_equi_state(self, state, i):
        """"""
        raise NotImplementedError("No implement process_data error!")

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

    def batch_forward(self, policy_value, states, masks, sub_batch=64):
        """分批次进行前向传播
        
        Args:
            policy_value (callable): 返回(old_probs, old_v)的函数
            states (Tensor): 完整状态张量 [N, state_dim]
            masks (Tensor): 完整掩码张量 [N, ...]
            sub_batch (int): 子批次大小
            
        Returns:
            (old_probs, old_v): 拼接后的完整结果
        """
        old_probs_list = []
        old_v_list = []
        
        total_size = len(masks)

        
        # 禁用梯度计算以节省显存
        with torch.no_grad():
            for start in range(0, total_size, sub_batch):
                end = min(start + sub_batch, total_size)
                
                # 提取子批次
                sub_states = tuple(arr[start:end] for arr in states)
                sub_masks = masks[start:end] if masks is not None else None
                # 前向传播
                sub_probs, sub_v = policy_value(sub_states, sub_masks)
                
                # 结果收集
                old_probs_list.append(sub_probs)
                old_v_list.append(sub_v)
        
        # 拼接结果
        return np.concatenate(old_probs_list, axis=0), np.concatenate(old_v_list, axis=0)

    def policy_update(self, is_ppo=False):
        """update the policy-value network using"""
        # print("buffer data: {}".format(self.buffer))
        buffer_list = list(self.buffer) # 转换为列表以便shuffle
        random.shuffle(buffer_list) # 混洗数据
        buffers = buffer_list # 更新 buffer (如果需要保持 buffer 类型)
        mini_batch = random.sample(buffers, self.batch_size)
        # print("mini batch is {}".format(mini_batch))
        state_batch = [data[0] for data in mini_batch]
        batch_state = self._prepare_batch_data(
            batch_state=state_batch, device=self.device
        )
        # batch_mask = np.array([(np.zeros_like(state.env.mask)[legal_action]=1 )for state in state_batch])
        batch_mask = np.array([
            np.where(np.isin(np.arange(state.env.mask.size), state.get_actions()), 1,0)
            for state in state_batch     
        ])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        value_batch = np.array([data[2] for data in mini_batch])
        for data in mini_batch:
            print(f"shape size is {len(data[3])}")
        advantages_batch = np.array([data[3] for data in mini_batch])
        action_batch = np.array([data[4] for data in mini_batch])

        policy_value = self.agent_net.policy_value
        reward_model = self.agent_net.reward_model

        mask_batch = np.array([s.mask.reshape(-1) for s in state_batch])
        old_probs, old_v = self.batch_forward(policy_value, batch_state, batch_mask)

        train_step = self.agent_net.train_step_pretrain
        if is_ppo == True:
            train_step = self.agent_net.train_step_cal

        for i in tqdm(range(self.epoches)):

            loss, value_loss, policy_loss, entropy = train_step(
                batch_state,
                batch_mask,
                mcts_probs_batch,
                value_batch,
                action_batch,
                advantages_batch,
                self.learn_rate * self.lr_multiplier,
                state_batch,
            )

            new_probs, new_v = self.batch_forward(policy_value, batch_state, batch_mask)
            # new_reward = reward_model(state_batch, action_batch)
            # new_probs = np.exp(new_probs)
            # new_v = new_v

            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            # kl = self.masked_kl_divergence(old_probs, new_probs, mask_batch)

            # print(
            #     "old v: {},\n new_v: {}, kl: {}".format(
            #         old_v, new_v, kl
            #     )
            # )
            if kl > self.kl_targ * 4:
                break

            if i+1 == self.epoches:
                self.kl_targ*=0.99
            #     tqdm.write(f"{i}-th kl targ is {self.kl_targ}")
        


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
        # print(f"value batcgh is {value_batch},  old v is {old_v}")

        # save step data
        self.kls.append(kl)
        self.explained_vars.append(explained_var_new)

        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        torch.cuda.empty_cache()
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

    def route_policy_update(self, is_ppo=False):
        mini_batch = random.sample(self.route_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        batch_state = self._prepare_batch_data(
            batch_state=state_batch, device=self.device
        )
        # batch_mask = np.array([state.env.route_mask.flatten() for state in state_batch])
        batch_mask = np.array([
            np.where(np.isin(np.arange(state.env.route_mask.size), state.get_route_actions()), 1, 0)
            for state in state_batch
            ])

        # print("state_batch: {}".format(state_batch))
        probs_batch = np.array([data[1] for data in mini_batch])
        value_batch = np.array([data[2] for data in mini_batch])
        advantages_batch = np.array([data[3] for data in mini_batch])
        action_batch = np.array([data[4] for data in mini_batch])

        policy_value = self.agent_net.route_policy_value
        mask_batch = np.zeros((len(state_batch), self.route_actions_size))
        for i in range(len(state_batch)):
            mask_batch[i] = state_batch[i].route_mask.reshape(-1)

        old_probs, old_v = policy_value(batch_state, batch_mask)
        # old_probs = np.exp(old_probs)
        print(f"old probs is {old_probs[old_probs>0]}, collect probs is {probs_batch[probs_batch>0]}")
        old_v = old_v

        train_route_step = self.agent_net.train_route_step
        if is_ppo:
            train_route_step = self.agent_net.train_route_step_ppo

        for i in tqdm(range(self.epoches)):
            loss, value_loss, policy_loss, entropy = train_route_step(
                batch_state,
                batch_mask,
                probs_batch,
                value_batch,
                action_batch,
                advantages_batch,
                self.learn_rate * self.route_lr_multiplier,
            )

            new_probs, new_v = policy_value(batch_state, batch_mask)
            # new_probs = np.exp(new_probs)
            new_v = new_v
            # print(
            #     "old_probs: {},\n new_probs: {}, kl: {}".format(
            #         old_probs, new_probs, kl
            #     )
            # )
            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )

            if kl > self.route_kl_targ * 4:
                break

        if kl > self.route_kl_targ * 2 and self.route_lr_multiplier > 0.1:
            self.route_lr_multiplier /= 1.5
        elif kl < self.route_kl_targ / 2 and self.route_lr_multiplier < 10:
            self.route_lr_multiplier *= 1.5
        # print(old_v.flatten())

        explained_var_old = 1 - np.var(
            np.array(value_batch) - old_v.flatten()
        ) / np.var(np.array(value_batch))
        explained_var_new = 1 - np.var(
            np.array(value_batch) - new_v.flatten()
        ) / np.var(np.array(value_batch))
        print("value batcgh is {value_batch},  old v is {old_v}")

        # save step data
        self.route_kls.append(kl)
        self.route_explained_vars.append(explained_var_new)
        self.route_policy_loss.append(policy_loss)
        self.route_value_loss.append(value_loss)

        tqdm.write(
            (
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.3f},"
                "explained_var_new:{:.3f}"
            ).format(
                kl,
                self.route_lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )
        torch.cuda.empty_cache()
        # self.agent_net.agentNet.eval()
        return loss, entropy

    def policy_evaluate(self, test_dfg, n_epoches=1):
        current_mcts_agent = MCTSPlayer(self.agent_net, 10)
        sum_latency = 0
        print("================policy evaluate=================")
        for i in range(n_epoches):
            # self.mapping.use_true_reward()
            mapping_task = Mapping(test_dfg, self.adg, self.operations)
            latency, data, route_data = mapping_task.start_mapping(current_mcts_agent)
            sum_latency -= latency
        return sum_latency / n_epoches
    
    def policy_evaluate_ppo(self, test_dfg, n_epoches=1):
        current_ppo_agent = MCTSPlayer(self.agent_net, 10)
        sum_latency = 0
        print("================policy evaluate=================")
        for i in range(n_epoches):
            # self.mapping.use_true_reward()
            mapping_task = Mapping(test_dfg, self.adg, self.operations)
            latency, data, route_data = mapping_task.grpo_mappingv2(current_ppo_agent)
            sum_latency += latency
        return sum_latency / n_epoches

    def save_figures(self):
        # 使用子图，避免多次调用 plt.figure()
        fig, axes = plt.subplots(
            nrows=2, ncols=5, figsize=(40, 8)
        )  # 创建10个子图，垂直排列

        # 绘制 loss
        axes[0, 0].plot(self.losses)
        axes[0, 0].set_ylabel("loss")
        axes[0, 0].set_title("Loss")

        # 绘制 latency (pref)
        axes[0, 1].plot(self.pref)
        axes[0, 1].set_ylabel("latency")
        axes[0, 1].set_title("Latency")

        # 绘制 kl
        axes[0, 2].plot(self.kls)
        axes[0, 2].set_ylabel("kl")
        axes[0, 2].set_title("KL")

        # 绘制 route kl
        axes[0, 3].plot(self.route_kls)
        axes[0, 3].set_ylabel("route kl")
        axes[0, 3].set_title("Route KL")

        # 绘制 explained_var
        axes[0, 4].plot(self.explained_vars)
        axes[0, 4].set_ylabel("explained_var")
        axes[0, 4].set_title("Explained Variance")

        # 绘制 route explained_var
        axes[1, 0].plot(self.route_explained_vars)
        axes[1, 0].set_ylabel("route explained_var")
        axes[1, 0].set_title("Route Explained Variance")

        # 绘制 route policy_loss
        axes[1, 1].plot(self.route_policy_loss)
        axes[1, 1].set_ylabel("route policy_loss")
        axes[1, 1].set_title("Route Policy Loss")

        # 绘制 route value_loss
        axes[1, 2].plot(self.latencies, marker='*')
        axes[1, 2].set_ylabel("all_latencies")
        axes[1, 2].set_title("all latencies")

        # 绘制 policy_loss
        axes[1, 3].plot(self.policy_loss)
        axes[1, 3].set_ylabel("policy_loss")
        axes[1, 3].set_title("Policy Loss")

        # 绘制 value_loss
        axes[1, 4].plot(self.value_loss)
        axes[1, 4].set_ylabel("value_loss")
        axes[1, 4].set_title("Value Loss")

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        plt.savefig("../figures/all_plots.png", dpi=600)  # 将所有子图保存到一个图片中
        plt.close(fig)  # 关闭整个figure
        self.save_data_txt()
    
    def save_data(self):
        data_to_save = {
            "loss": self.losses,
            "latency": self.pref,
            "kl": self.kls,
            "route_kl": self.route_kls,
            "explained_var": self.explained_vars,
            "route_explained_var": self.route_explained_vars,
            "route_policy_loss": self.route_policy_loss,
            "all_latencies": self.latencies,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
        }

        output_dir = "../figures/data/"
        os.makedirs(output_dir, exist_ok=True)

        for key, data_list in data_to_save.items():
            filename = os.path.join(output_dir, f"{key}.csv")
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([key])  # 标题行
                for item in data_list:
                    writer.writerow([item])
            print(f"数据 '{key}' 已保存到 {filename}")
    
    def save_data_txt(self):
        data_to_save = {
            "loss": self.losses,
            "latency": self.pref,
            "kl": self.kls,
            "route_kl": self.route_kls,
            "explained_var": self.explained_vars,
            "route_explained_var": self.route_explained_vars,
            "route_policy_loss": self.route_policy_loss,
            "all_latencies": self.latencies,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
        }

        output_dir = "../figures/data/"
        os.makedirs(output_dir, exist_ok=True)

        for key, data_list in data_to_save.items():
            filename = os.path.join(output_dir, f"{key}.txt")  # 可以保存为 .txt 或 .csv
            data_array = np.array(data_list)

            # 保存为一列的文本文件
            np.savetxt(filename, data_array, delimiter=',', header=key, comments='')
            print(f"数据 '{key}' 已使用 numpy.savetxt 保存到 {filename}")

    def run(self, dfgs, is_multi=False):
        """a train pipeline for mapping"""
        try:
            test_dfg = DFGParser("dfg.json").getDFG()

            for i in tqdm(range(self.batch_num), position=-1):

                if i % 3 == 0:
                    if is_multi:
                        self.clean_data()
                        self.collect_data_multithread(1, dfgs, 8)
                        self.agent.step()
                    else:
                        self.clean_data()
                        collect_data_multi(self, 1, dfgs, 12)
                        self.agent.step()
                tqdm.write(f"the collect data buffer size is {len(self.buffer)}")
                if len(self.buffer) >= self.batch_size:
                    loss, _ = self.policy_update()
                    self.losses.append(loss)
                if (i + 1) % self.check_freq == 0:
                    print(f"current batch: {i+1}")
                    torch.save(self.agent_net.agentNet, "../models/cur-agent.pt")
                    # TODO: fix the policy_value function
                    cur_latency = self.policy_evaluate(test_dfg)
                    print(f"current latency: {cur_latency}")
                    if cur_latency is None:
                        cur_latency = -500
                    self.pref.append(cur_latency)
                    if cur_latency < self.best_latency:
                        self.best_latency = cur_latency
                        torch.save(self.agent_net.agentNet, "../models/best-agent.pt")

                if i % 5 == 0:
                    self.save_figures()

            self.save_figures()

            with open("../models/mcts.pkl", "wb") as f:
                pickle.dump(self.mcst, f)

        except KeyboardInterrupt:
            print("\n\rquit")

    def run_ppo(self, dfgs, is_multi=False):
        """a ppo train pipeline for mapping"""
        try:
            test_dfg = DFGParser("../example/dfg.json").getDFG()

            for i in tqdm(range(self.batch_num), position=-1):

                if i % 3 == 0:
                    if is_multi:
                        self.collect_data_multithread(1, dfgs, 8)
                        # self.agent.step()
                    else:
                        collect_data_multi(self, 1, dfgs, 8)
                        # self.agent.step()
                tqdm.write(f"the collect data buffer size is {len(self.buffer)}")
                if len(self.buffer) >= self.batch_size:
                    loss, _ = self.policy_update(is_ppo=True)
                    self.losses.append(loss)
                    if i % 3 == 2:
                        self.clean_data()

                if (i + 1) % self.check_freq == 0:
                    print(f"current batch: {i+1}")
                    torch.save(self.agent_net.agentNet, "../models/cur-agent-ppo.pt")
                    # TODO: fix the policy_value function
                    cur_latency = self.policy_evaluate_ppo(test_dfg)
                    print(f"current latency: {cur_latency}")
                    if cur_latency is None:
                        cur_latency = -500
                    self.pref.append(cur_latency)
                    if cur_latency < self.best_latency:
                        self.best_latency = cur_latency
                        torch.save(
                            self.agent_net.agentNet, "../models/best-agent_ppo.pt"
                        )

            
                self.save_figures()

            self.save_figures()

            with open("../models/mcts.pkl", "wb") as f:
                pickle.dump(self.mcst, f)

        except KeyboardInterrupt:
            print("\n\rquit")