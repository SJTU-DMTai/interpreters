# interpreters of stock graph

## 数据获取

## 获取解释
给定股票和时间获取解释，参考`src/run_explain.py`中的`run_one_test()`。

## 测试
- 测试脚本
	- 直接运行`./eval_explanation.sh`
	- `src/run_expain.py` 为测试入口，其输入参数为：
	    - `--data_root` 为存放股票数据（包括图数据和价格数据）的路径 ，注意最好使用绝对路径；
	    - `--ckpt_root` 为存放模型参数的路径，如 `"/home/jiale/tmp_ckpt/"` ，注意最好使用绝对路径；
	    - `--result_root` 为存放解释结果的路径，如 `"/home/jiale/results/"` ，注意最好使用绝对路径；
	    - `--market` 为所需要测试的股票市场名称，支持 `"A_share"` （cn_data）；
	    - `--relation_type` 为所需要测试的关系图的类型，支持 `"stock-stock"`；
	    - `--graph_model` 为需要测试的图神经网络模型的类型，支持`"RSR"`、`"GAT"`以及`"simpleHGN"`；
	    - `--graph_type` 为需要测试的图是否同构，支持 `"heterograph"`（异构图）以及`"homograph"`（同构图）。

- 解释形式
  
    解释默认存储在`result_root`目录下，使用python pickle可以打开文件，得到一个python字典，其中的“explanation”字段对应了解释，下面对解释的形式进行说明。
    
    - 打开方式：
      
        ```python
        import pickle
        import os
        
        # effect解释
        with open(os.path.join(args.result_root, f"{args.market}-{args.graph_model}-{args.graph_type}-att-explanation"), 'rb') as f:
        		exp_att = pickle.load(f)
        
        # SubgraphX解释
        with open(os.path.join(args.result_root, f"{args.market}-{args.graph_model}-{args.graph_type}-subgraphx-explanation"), 'rb') as f:
        		exp_xpath = pickle.load(f)
        
        # xPath解释
        with open(os.path.join(args.result_root, f"{args.market}-{args.graph_model}-{args.graph_type}-xpath-explanation"), 'rb') as f:
        		exp_xpath = pickle.load(f)
        ```
        
    - effect解释：对应`exp_att["explanation"]`下的某一个字段
      
        ```bash
        "SZ300251": [
        			[261, 0.1845821738243103],
        			[298, 0.16644789278507233],
        			[238, 0.13847197592258453],
        			[278, 0.11163753271102905],
        			[284, 0.10435605049133301],
        			[277, 0.09626047313213348]
        		],
        ```
        
        对应股票SZ300251（图中的第298号节点）每个邻居节点的重要性分数，以`[261, 0.1845821738243103]` 为例，它表示第261号股票对应的重要性分数为0.1845821738243103。
        
    - SubgraphX解释：对应`exp_xpath["explanation"]`下的某一个字段
      
        ```python
        "SZ300251": [
        	[[298, 261, 277, 278, 284], 0.01779908500611782],
        	[[298, 238, 261, 277, 284], 0.009899597615003586],
        	......
        ]
        ```
        
        对应SZ300251（图中的第298号节点）的邻居节点所构成子图的重要性，以`[[298, 261, 277, 278, 284], 0.01779908500611782]`为例，`[298, 261, 277, 278, 284]`代表由这些节点构成的子图，`0.01779908500611782`代表这个子图的重要性分数。
        
    - xPath解释：对应`exp_xpath["explanation"]`下的某一个字段
      
        ```bash
        "SZ300251": {
        	(298, 261): 532520.0299918652,
        	(298, 238): 785947.1921622753,
        	(298, 278): 57.33989179134369,
        	(298, 284): 302744.3151175976,
        	(298, 277): 345110.0568473339
        }
        ```
        
        对应股票SZ300251（图中的第298号节点）对应的每条边的重要性分数，以`(298, 261): 532520.0299918652` 为例，它表示从298出发指向261这条边对应的评分为532520.0299918652。
    
- 解释指标：
    - 可信度（fidelity）分数是评估可解释方法性能的主要指标，它表示在仅保留解释情况下的输出和原始输出的相似程度，fidelity分数越低，则表明该解释方法找到的解释越能还原模型的预测。同时，我们还引入了稀疏度（sparsity）的度量，我们认为一个好的解释应该是较为稀疏的，我们应该尽量用较少的输入信息去解释模型，这样能够保证解释便于人理解。
    - 我们在A股CSI300的基础上进行测试，所分别选取三个待解释模型（RSR、SimpleHGN和GAT）以及三个解释方法（effect、subgraphX和xPath），测试了在不同sparsity下的fidelity指标，以及不同解释方法生成一次解释所需要的时间。