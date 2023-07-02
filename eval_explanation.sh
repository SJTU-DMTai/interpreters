
# git clone https://github.com/SJTU-Quant/qlib.git
#cd qlib
# python setup.py develop

#cd examples/benchmarks/Graphs_dgl

echo "============================= 对RSR的预测结果进行解释 ============================="
python run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
                      --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
                      --result_root "/data/dengjiale/qlib_exp/results/" \
                      --market "A_share" \
                      --relation_type "stock-stock" \
                      --graph_model "RSR" \
                      --graph_type "heterograph" \
                      --gpu 0

echo "============================= 对simpleHGN的预测结果进行解释 ============================="
python run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
                     --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
                     --result_root "/data/dengjiale/qlib_exp/results/" \
                     --market "A_share" \
                     --relation_type "stock-stock" \
                     --graph_model "simpleHGN" \
                     --graph_type "heterograph" \
                     --gpu 0

echo "============================= 对GAT的预测结果进行解释 ============================="
python run_ts_dgl.py --data_root "/home/zhangzexi/.qlib/qlib_data/" \
                      --ckpt_root "/data/dengjiale/qlib_exp/tmp_ckpt/" \
                      --result_root "/data/dengjiale/qlib_exp/results/" \
                      --market "A_share" \
                      --relation_type "stock-stock" \
                      --graph_model "GAT" \
                      --graph_type "homograph" \
                      --gpu 0