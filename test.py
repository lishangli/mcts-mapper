from adgParser import *
from dfgParser import *
from model import *
from operations import Operations
from mapping import Mapping
from mcts import MCTSPlayer

# 使用示例
dfg_parser = DFGParser('dfg.json')
adg_parser = ADGIR('cgra_adg.json')
operations = Operations()
operations.OpParser("operations.json")
# s
dfg = dfg_parser.getDFG()
adg = adg_parser.getADG()
# env = Environment(dfg, adg, operations)
# state = MappingState(env)
# state.draw()

class MappingTest(object):
    def __init__(self,dfg, adg, operations):
        self.dfg = dfg
        self.adg = adg
        self.operations = operations
        self.mapping = Mapping(dfg, adg, operations)

    def test(self, model_file):
        actions_len = len(self.dfg.getNodes()) * (self.adg.getMaxNodeId()+1)
        agentNet = AgentNetwork(7, 5, 32, 3230,32,use_gpu = True, model_file=model_file)
        agent = MCTSPlayer(agentNet)
        # mapping = Mapping(self.dfg, adg,operations)

        self.mapping.test_mapping(agent, is_shown=True, temp=1e-2)

mt = MappingTest(dfg, adg, operations)
mt.test("models/best-agent.pt")