import argparse
import torch
import pickle
import os
import time

from interpreter.attentionx import AttentionX
from interpreter.xpath import xPath
from interpreter.subgraphx import SubgraphXExplainer
from utils import setup_seed, get_model, get_dataset


def eval_explanation(explainer, explainer_name, sparsity, step_size):
    res_path = os.path.join(args.result_root,
                            f"{args.market}-{args.graph_model}-{args.graph_type}-{explainer_name}-explanation")
    if os.path.exists(res_path):
        with open(res_path, 'rb') as f:
            explanation = pickle.load(f)
    else:
        explanation, scores = model.get_explanation(dataset, explainer, debug=True, step_size=step_size)
        exp = {}
        exp['explainer'] = explainer_name
        exp['explanation'] = explanation
        exp['scores'] = scores
        print('Saving explanations...')
        with open(res_path, 'wb') as f:
            pickle.dump(exp, f)

    res_explainer = {}
    for i, k in enumerate(sparsity[args.graph_model][explainer_name]):
        _, fidelity = model.get_explanation(dataset, explainer, step_size=step_size, top_k=k,
                                            cached_explanations=explanation['explanation'])
        res_explainer[i+3] = sum(fidelity) / len(fidelity)
    return res_explainer


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


def run_all_test():
    # hyper-parameters
    sparsity = {
        'RSR': {'effect': [5, 9], 'xpath': [3, 5], 'subgraphx': [4, 8]},
        'simpleHGN': {'effect': [3, 5], 'xpath': [3, 5], 'subgraphx': [4, 8]},
        'GAT': {'effect': [3, 5], 'xpath': [3, 5], 'subgraphx': [3, 6]},
    }

    graph_type = args.graph_type  # for GAT, use 'homograph'
    attn_explainer = AttentionX(graph_model=graph_type, num_layers=1, device=device)
    xpath_explainer = xPath(graph_model=graph_type, num_layers=1, device=device)
    subagraphx_explainer = SubgraphXExplainer(graph_model=graph_type, num_layers=1, device=device)

    res = {}
    res['effect'] = eval_explanation(attn_explainer, 'effect', sparsity, 100)
    res['xpath'] = eval_explanation(xpath_explainer, 'xpath', sparsity, 100)
    res['subgraphx'] = eval_explanation(subagraphx_explainer, 'subgraphx', sparsity, 200)

    print('以下是测试结果：')
    print('>>> 当解释大小限定为3时，各解释方法的平均fidelity分数分别为 <<<')
    print(f'effect: \t{round(res["effect"][3], 6)}')
    print(f'xpath: \t\t{round(res["xpath"][3], 6)}')
    print(f'subgraphx: \t{round(res["subgraphx"][3], 6)}')
    print('>>> 当解释大小限定为4时，各解释方法的平均fidelity分数分别为 <<<')
    print(f'effect: \t{round(res["effect"][4], 6)}')
    print(f'xpath: \t\t{round(res["xpath"][4], 6)}')
    print(f'subgraphx: \t{round(res["subgraphx"][4], 6)}')


def run_one_test():
    '''
    get explanations for stocks in a given time period
    '''
    stocks = ['SH600000','SH600009' ]
    # stocks = None
    start_time = '2021-01-03'
    end_time = '2021-01-08'

    xpath_explainer = xPath(graph_model=args.graph_type, num_layers=1, device=device)
    attn_explainer = AttentionX(graph_model=args.graph_type, num_layers=1, device=device)
    # subagraphx_explainer = SubgraphXExplainer(graph_model=args.graph_type, num_layers=1, device=device)
    t1 = time.time()
    explanation = model.get_explanation(dataset, stocks=stocks, start_time=start_time, end_time=end_time, explainer=xpath_explainer, top_k=3)
    print(f'xpath explanation time: {time.time() - t1}')
    return explanation


if __name__ == '__main__':
    setup_seed(2023)
    args = parse_args()
    device = torch.device("cuda:%d" % (args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else "cpu")

    dataset = get_dataset(args.data_root)
    model = get_model(args, dataset, device)

    run_one_test()
    # run_all_test()

