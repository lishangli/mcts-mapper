"""MAIN FUNCTION ENTRY"""
from trainner import MCTS_RL
from models import AgentNetwork
from utils import JSONDataset
from parser import DFGParser, ADGIR, Operations, DFGFeatures, ADGFeatures

if __name__ == "__main__":

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

    # model = GAT(nfeat=7, nhid=8, nclass=7, dropout=0.6, alpha=0.2, nheads=8)
    features = DFGFeatures(dfg).getFeaturesVector(operations)
    adg_features = ADGFeatures(adg).getFeaturesVector()
    actions_len = adg.getNodeNums() * (adg.getMaxNodeId() + 1)
    route_size = len(adg.getEdges())
    print(f"action len is {actions_len}")
    route_size = len(adg.getEdges())
    agent_net = AgentNetwork(
        7,
        5,
        32,
        actions_len,
        route_size,
        32,
        True,
        model_file=None
    )
    data_loader = JSONDataset("./dataset/microbench/json", operations, transform=None)
    dfgs = []
    for data in data_loader:
        dfg = data["dfg"]
        dfgs.append(dfg)
    rl = MCTS_RL(adg,operations, agent_net)
    rl.run_ppo(dfgs, is_multi=False)