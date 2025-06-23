from adgParser import *
from dfgParser import *
from mcts import MCTSPlayer
from net import AgentNetwork
from operations import Operations
from mapping import Mapping
from mcts import MCTSPlayer
from ppo import PPOPlayer
from line_profiler import profile

# 使用示例
dfg_parser = DFGParser("dfg.json")
adg_parser = ADGIR("cgra_adg.json")
operations = Operations()
operations.OpParser("operations.json")
# s
dfg = dfg_parser.getDFG()
adg = adg_parser.getADG()
# env = Environment(dfg, adg, operations)
# state = MappingState(env)
# state.draw()


class MappingTest(object):
    def __init__(self, dfg, adg, operations, model_file=None):
        self.dfg = dfg
        self.adg = adg
        self.operations = operations
        route_size = len(self.adg.getEdges())
        actions_len = self.adg.getNodeNums() * (self.adg.getMaxNodeId() + 1)
        agentNet = AgentNetwork(
            7,
            5,
            32,
            actions_len,
            route_size,
            32,
            use_gpu=True,
            model_file=model_file,
        )
        self.agent = MCTSPlayer(agentNet, 10)
        self.ppo_agent = PPOPlayer(agentNet)
        self.mapping = Mapping(dfg, adg, operations, False)

    @profile
    def test(self):
        # mapping = Mapping(self.dfg, adg,operations)
        self.mapping.use_true_reward()
        # self.agent.t += 1
        print("12")
        latency = self.mapping.test_mapping(self.agent, is_shown=True, temp=1e-2)
        print("test {}".format(latency))

    def ppo_test(self):
        self.mapping.use_true_reward()
        latency = self.mapping.ppo_mapping(self.agent, is_shown=True)
        print("test {}".format(latency))


if __name__ == "__main__":
    mt = MappingTest(dfg, adg, operations, "models/best-agent.pt")
    mt.test()
