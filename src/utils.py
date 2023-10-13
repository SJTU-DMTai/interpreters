import torch
import numpy as np
import random
import logging
import os
import pickle
from graph_data import stock_stock_data, stock_concept_data
from models.ts_model import Graphs
from dataset.dataset import DatasetH


def load_graph(market, relation_source, data_mix, data_root):
    indexs = data_mix.index.levels[1].tolist()
    indexs = list(set(indexs))
    stocks_sorted_list = sorted(indexs)
    # print(stocks_sorted_list)
    # print("number of stocks: ", len(stocks_sorted_list))
    stocks_index_dict = {}
    for i, stock in enumerate(stocks_sorted_list):
        stocks_index_dict[stock] = i
    n = len(stocks_index_dict.keys())
    if relation_source == 'stock-stock':
        rel_encoding = stock_stock_data.get_all_matrix(
            market, stocks_index_dict,
            # data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/"
            data_path=os.path.join(data_root, 'graph_data')  # This is the graph_data path on the server
        )
        return rel_encoding, stocks_sorted_list
    elif relation_source == 'industry':
        industry_dict = stock_concept_data.read_graph_dict(
            market,
            relation_name="SW_belongs_to",
            # data_path="D:/Code/myqlib/.qlib/qlib_data/graph_data/",
            data_path=os.path.join(data_root, 'graph_data')
        )
        return stock_concept_data.get_full_connection_matrix(
            industry_dict, stocks_index_dict
        ), stocks_sorted_list
    elif relation_source == 'full':
        return np.ones(shape=(n, n)), stocks_sorted_list
    else:
        raise ValueError("unknown graph name `%s`" % relation_source)


def init_logger(log_dir, args):
    os.makedirs(log_dir, exist_ok=True)
    # current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_file = log_dir + f'/{args.graph_model}.log'
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger('updateSecurity')
    logger.setLevel('INFO')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_model(args, dataset, device):
    with open(
            f'{args.data_root}/dataframe_mix_csi300_rankTrue_alpha360_horizon1.pkl',
            'rb') as f:
        data_mix= pickle.load(f)
    rel_encoding, stock_name_list = load_graph(args.market, args.relation_type, data_mix['data'], args.data_root)
    model = Graphs(graph_model=args.graph_model,  # 'GAT' or 'simpleHGN', 'RSR'
                   d_feat=6, hidden_size=64, num_layers=2, loss="mse", dropout=0.7, n_epochs=100,
                   metric="loss", base_model="LSTM", use_residual=True, GPU=args.gpu, lr=1e-3,
                   early_stop=10, rel_encoding=rel_encoding, stock_name_list=stock_name_list,
                   num_graph_layer=2, logger=init_logger('../log', args))
    model.to(device)
    model_path = os.path.join(args.ckpt_root, f"{args.market}-{args.graph_model}-{args.graph_type}.pt")
    if os.path.exists(model_path):
        model.load_checkpoint(model_path)
        # df_test = dataset.prepare("test", col_set=["feature", "label"], data_key="infer")
        # x_test, y_test = df_test["feature"], df_test["label"]
        # print("evaluating...")
        # test_loss, test_IC, test_RIC = model.test_epoch(x_test, y_test)
        # print("test loss %.4f, test IC %.4f, test RIC %.4f" % (test_loss, test_IC, test_RIC))
    else:
        model.fit(dataset, save_path=model_path)

    return model


def get_dataset(data_root):
    train_start_date = '2008-01-01'
    train_end_date = '2014-12-31'
    valid_start_date = '2015-01-01'
    valid_end_date = '2016-12-31'
    test_start_date = '2017-01-01'
    test_end_date = '2022-12-31'

    with open(f'{data_root}/dataframe_mix_csi300_rankTrue_alpha360_horizon1.pkl', 'rb') as f:
        df_mix = pickle.load(f)
    dataset = DatasetH(df_mix, train_start_date, train_end_date, valid_start_date,
                       valid_end_date, test_start_date, test_end_date)
    # df_test = dataset.prepare("test", col_set=["feature", "label"])
    return dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False