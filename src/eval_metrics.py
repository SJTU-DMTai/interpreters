import argparse
import torch
import numpy as np
from utils import setup_seed, get_model, get_dataset
from interpreter.xpath import xPath
from sklearn.metrics import r2_score


class ReliabilityMetrics():
    def __init__(self, model, explainer, data):
        self.model = model
        self.explainer = explainer
        self.data = data
        self.date_range = None
        self.stock_list = None

    def set_selection(self, date_range, stock_list):
        self.date_range = date_range
        self.stock_list = stock_list

    def dependability(self, ):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Explanation evaluation.")
    parser.add_argument("--data_root", type=str, default="/home/jiale/.qlib/qlib_data/", help="graph_data root path")
    parser.add_argument("--ckpt_root", type=str, default="/home/jiale/interpreters/tmp_ckpt/", help="ckpt root path")
    parser.add_argument("--result_root", type=str, default="/home/jiale/interpreters/results/",
                        help="explanation resluts root path")
    parser.add_argument("--market", type=str, default="A_share",
                        choices=["A_share"], help="market name")
    parser.add_argument("--relation_type", type=str, default="stock-stock",
                        choices=["stock-stock", "industry", "full"], help="relation type of graph")
    parser.add_argument("--graph_model", type=str, default="RSR",
                        choices=["RSR", "GAT", "GCN", "simpleHGN",
                                  "GSAT_RSR", "GSAT_GAT", "GSAT_GCN",
                                  "SNex_RSR", "SNex_GAT", "SNex_GCN"], help="graph moddel name")
    parser.add_argument("--graph_type", type=str, default="heterograph",
                        choices=["heterograph", "homograph"], help="graph type")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    setup_seed(2023)
    args = parse_args()
    device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    dataset = get_dataset(args.data_root)
    model = get_model(args, dataset, device)
    xpath_explainer = xPath(graph_model=args.graph_type, num_layers=1, device=device)

    rm = ReliabilityMetrics(model, xpath_explainer, dataset)

    stocks = ['SH600000', 'SH600009']
    # stocks = None
    start_time = '2021-01-03'
    end_time = '2021-01-08'

    rm.set_selection((start_time, end_time), stocks)

