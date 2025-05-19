"""MAIN FUNCTION ENTRY"""
from trainner import MCTS_RL
import trainner as tr
from models import AgentNetwork
from utils import JSONDataset
from parser import ADGIR, Operations, ADGFeatures

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments Settings")

    parser.add_argument("--dfg_path", type=str, default="../dataset/microbench/json", help="Dataflow graph Path(default:./dataset/microbench/json)")
    parser.add_argument("--adg_path", type=str, default="../example/cgra_adg.json", help="Architechture Descripthion Path(default:../example/cgra_adg.json)")
    parser.add_argument("--operations_path", type=str, default= "../example/operations.json",help="Operations Path(default:../example/operations.jsonn)")

    args = parser.parse_args()


    adg_parser = ADGIR(args.adg_path)
    
    operations = Operations()
    operations.OpParser(args.operations_path)
    
    data_loader = JSONDataset(args.dfg_path, operations, transform=None)
    name2op = operations.name2op
    adg = adg_parser.getADG()
    # if torch.cuda.is_available():
    #     print("CUDA is available. You can use GPU.")
    # else:
    #     print("CUDA is not available. Please check your CUDA installation.")

    # model = GAT(nfeat=7, nhid=8, nclass=7, dropout=0.6, alpha=0.2, nheads=8)
    # adg_features = ADGFeatures(adg).getFeaturesVector()
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
    dfgs = []
    for data in data_loader:
        dfg = data["dfg"]
        dfgs.append(dfg)
    rl = MCTS_RL(adg,operations, agent_net)
    rl.run_ppo(dfgs, is_multi=False)