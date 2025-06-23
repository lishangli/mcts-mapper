"""MAIN FUNCTION ENTRY"""

from trainner import MCTS_RL
import trainner as tr
from models import AgentNetwork
from utils import JSONDataset
from parser import ADGIR, Operations, ADGFeatures

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments Settings")

    parser.add_argument(
        "--dfg_path",
        type=str,
        default="../dataset/microbench/json",
        help="Dataflow graph Path(default:./dataset/microbench/json)",
    )
    parser.add_argument(
        "--adg_path",
        type=str,
        default="../example/cgra_adg.json",
        help="Architechture Descripthion Path(default:../example/cgra_adg.json)",
    )
    parser.add_argument(
        "--operations_path",
        type=str,
        default="../example/operations.json",
        help="Operations Path(default:../example/operations.jsonn)",
    )
    parser.add_argument("--batch_num", type=int, default=800, help="Number of batches")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of each batch")
    parser.add_argument("--kl_targ", type=float, default=0.1, help="KL target value")
    parser.add_argument(
        "--route_kl_targ", type=float, default=0.1, help="Route KL target value"
    )
    parser.add_argument("--check_freq", type=int, default=5, help="Check frequency")
    parser.add_argument(
        "--target_latency", type=int, default=100, help="Target latency"
    )
    parser.add_argument("--best_latency", type=int, default=1000, help="Best latency")
    parser.add_argument(
        "--lr_multiplier", type=float, default=1.0, help="Learning rate multiplier"
    )
    parser.add_argument(
        "--route_lr_multiplier",
        type=float,
        default=1.0,
        help="Route learning rate multiplier",
    )
    parser.add_argument("--learn_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epoches", type=int, default=50, help="Number of epochs")
    parser.add_argument("--state_size", type=int, default=32, help="State size")

    # MCTS configuration
    parser.add_argument(
        "--discount", type=float, default=0.99, help="RL reward discount"
    )
    parser.add_argument(
        "--c_param", type=float, default=4, help="MCTS exploration parameter"
    )
    parser.add_argument(
        "--expand_size", type=int, default=20, help="MCTS expand nodes size"
    )

    args = parser.parse_args()

    adg_parser = ADGIR(args.adg_path)

    operations = Operations()
    operations.OpParser(args.operations_path)

    # Parse the architecture description and operations files to initialize the ADG and operations objects.

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

    route_size = len(adg.getEdges())
    agent_net = AgentNetwork(
        7, 5, 32, actions_len, route_size, 32, True, model_file=None
    )
    dfgs = []
    for data in data_loader:
        dfg = data["dfg"]
        dfgs.append(dfg)
    rl = MCTS_RL(adg, operations, agent_net, parser.parse_args())
    rl.run_ppo(dfgs, is_multi=False)
