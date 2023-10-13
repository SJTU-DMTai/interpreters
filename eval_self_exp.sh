cd src

#echo "============================= RSR ============================="
#python run_explain.py --graph_model "RSR"  --graph_type "heterograph"
#
#echo "============================= GAT ============================="
#python run_explain.py --graph_model "GAT" --graph_type "homograph"
#
#echo "============================= GCN ============================="
#python run_explain.py --graph_model "GCN" --graph_type "homograph"
#
#
#echo "============================= GSAT_RSR ============================="
#python run_explain.py --graph_model "GSAT_RSR"  --graph_type "heterograph"
#
#echo "============================= GSAT_GAT ============================="
#python run_explain.py --graph_model "GSAT_GAT" --graph_type "homograph"
#
#echo "============================= GSAT_GCN ============================="
#python run_explain.py --graph_model "GSAT_GCN" --graph_type "homograph"
#
#
#echo "============================= SNex_RSR ============================="
#python run_explain.py --graph_model "SNex_RSR"  --graph_type "heterograph"
#
#echo "============================= SNex_GAT ============================="
#python run_explain.py --graph_model "SNex_GAT" --graph_type "homograph"

echo "============================= SNex_GCN ============================="
python run_explain.py --graph_model "SNex_GCN" --graph_type "homograph"
