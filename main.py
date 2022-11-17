from train import *
from rdkit import Chem
import pandas as pd
import wandb
import os
if __name__ == '__main__':
    sweep_config = {
            "name": "graphrnn-v2-sweep",
            "metric": {"name": "Val Atomic Number Accuracy", "goal": "maximize"},
            "method": "random",
            "parameters": {
                "hidden_size_rnn": {'values': [8, 16, 32, 64, 128, 256, 512, 1025]},

                "embedding_size_rnn": {'values': [32, 64, 128, 256, 512, 1024]},

                "num_layers": {'max': 10, 'min': 2},

                'lr': {'max': 0.003, 'min': 0.0000005},

                "output_lr": {'max': 0.003, 'min': 0.0000005},

                # "node_lr": {'max': 0.003, 'min': 0.0000005},
                "feature_embedding": {'values': [32, 64, 128, 256, 512, 1024]},

                "feature_out": {'values': [8,16,32, 64, 128, 256, 512, 1024]},

                'atomic_lr': {'max': 0.005, 'min': 0.000005},

                'formal_lr': {'max': 0.005, 'min': 0.000000005},

                'chiral_lr': {'max': 0.005, 'min': 0.000000005},

                'hybr_lr': {'max': 0.005, 'min': 0.000000005},

                'numh_lr': {'max': 0.005, 'min': 0.000000005},

                'aromatic_lr': {'max': 0.005, 'min': 0.000000005},
                'atomic_loss_factor': {'min':1.0, 'max':3.0}
            }
    }
    def main():
        wandb.init(project="GraphRNN-v2")
        # wandb.init(project="GraphRNN", mode="disabled")
        # All necessary arguments are defined in args.py
        # config = dict(wandb.config)
        # args = Args(my_dict=config)
        args = Args()
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        # print('CUDA', args.cuda)
        print('File name prefix',args.fname)
        # check if necessary directories exist
        if not os.path.isdir(args.model_save_path):
            os.makedirs(args.model_save_path)
        if not os.path.isdir(args.graph_save_path):
            os.makedirs(args.graph_save_path)
        if not os.path.isdir(args.figure_save_path):
            os.makedirs(args.figure_save_path)
        if not os.path.isdir(args.timing_save_path):
            os.makedirs(args.timing_save_path)
        if not os.path.isdir(args.figure_prediction_save_path):
            os.makedirs(args.figure_prediction_save_path)
        if not os.path.isdir(args.nll_save_path):
            os.makedirs(args.nll_save_path)

        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
        # if args.clean_tensorboard:
        #     if os.path.isdir("tensorboard"):
        #         shutil.rmtree("tensorboard")
        # configure("tensorboard/run"+time, flush_secs=5)

        # https://github.com/dakoner/keras-molecules/blob/master/convert_rdkit_to_networkx.py
        
        zinc_df = pd.read_csv('./SMILES_Keyboard.csv')
        zinc_df = zinc_df.loc[zinc_df['SMILES'].str.len()<50][['SMILES', 'NAME']].reset_index(drop=True)
        graphs = []
        max_len = 0
        for smStr in zinc_df["SMILES"]:
            # print(dir(torch_geometric.utils))
            mol = Chem.MolFromSmiles(smStr)
            graphs.append(mol_to_nx(mol))
            # print(nx_to_mol(mol_to_nx(mol)))
        # graphs = create_graphs.create(args)
        
        # split datasets
        random.seed(123)
        shuffle(graphs)
        graphs_len = len(graphs)
        graphs_test = graphs[int(0.7 * graphs_len):]
        graphs_train = graphs[0:int(0.7*graphs_len)]
        graphs_validate = graphs_train[0:int(0.3*len(graphs_train))]
        graphs_train =  graphs_train[int(0.3*len(graphs_train)):]
        # if use pre-saved graphs
        # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
        # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
        #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
        # graphs = load_graph_list(fname_test, is_real=True)
        # graphs_test = graphs[int(0.8 * graphs_len):]
        # graphs_train = graphs[0:int(0.8 * graphs_len)]
        # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]


        graph_validate_len = 0
        for graph in graphs_validate:
            graph_validate_len += graph.number_of_nodes()
        graph_validate_len /= len(graphs_validate)
        print('graph_validate_len', graph_validate_len)

        graph_test_len = 0
        for graph in graphs_test:
            graph_test_len += graph.number_of_nodes()
        graph_test_len /= len(graphs_test)
        print('graph_test_len', graph_test_len)



        args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
        max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
        min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

        # args.max_num_node = 2000
        # show graphs statistics
        print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
        print('max number node: {}'.format(args.max_num_node))
        print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
        print('max previous node: {}'.format(args.max_prev_node))

        # save ground truth graphs
        ## To get train and test set, after loading you need to manually slice
        save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
        save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
        print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

        ### comment when normal training, for graph completion only
        # p = 0.5
        # for graph in graphs_train:
        #     for node in list(graph.nodes()):
        #         # print('node',node)
        #         if np.random.rand()>p:
        #             graph.remove_node(node)
            # for edge in list(graph.edges()):
            #     # print('edge',edge)
            #     if np.random.rand()>p:
            #         graph.remove_edge(edge[0],edge[1])

        ### dataset initialization
        if 'nobfs' in args.note:
            print('nobfs')
            dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
            args.max_prev_node = args.max_num_node-1
        if 'barabasi_noise' in args.graph_type:
            print('barabasi_noise')
            dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
            args.max_prev_node = args.max_num_node - 1
        else:
            args.max_prev_node = args.max_num_node - 1
            dataset = Graph_sequence_sampler_pytorch(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
        sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                         num_samples=args.batch_size*args.batch_ratio, replacement=True)
        dataset_val = Graph_sequence_sampler_pytorch(graphs_validate,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                   sampler=sample_strategy)
        
        val_dataset_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers)
        ### model initialization
        ## Graph RNN VAE model
        # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
        #                   hidden_size=args.hidden_size, num_layers=args.num_layers).to(args.device)
        NUM_POS_ATOMS = 118
        NUM_POS_EDGES = len(Chem.rdchem.BondType.names.keys())
        if 'GraphRNN_VAE_conditional' in args.note:
            rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                            has_output=False, device=args.device).to(args.device)
            output = MLP_VAE_conditional_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).to(args.device)
        elif 'GraphRNN_MLP' in args.note:
            rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                            has_output=False,device=args.device).to(args.device)
            output = MLP_plain(h_size=args.hidden_size_rnn, embedding_size=args.embedding_size_output, y_size=NUM_POS_EDGES, max_node=dataset.max_prev_node).to(args.device)
            node_layers = []
            node_layers.append(MLP_plain_regress(h_size=args.feature_out, embedding_size=args.embedding_size_output,y_size=NUM_POS_ATOMS).to(args.device))
            node_layers.append(MLP_plain_regress(h_size=args.feature_out, embedding_size=args.embedding_size_output, y_size=1).to(args.device))
            node_layers.append(MLP_plain_regress(h_size=args.feature_out, embedding_size=args.embedding_size_output, y_size=len(Chem.rdchem.ChiralType.values.keys())).to(args.device))
            node_layers.append(MLP_plain_regress(h_size=args.feature_out, embedding_size=args.embedding_size_output, y_size=len(Chem.rdchem.HybridizationType.values.keys())).to(args.device))
            node_layers.append(MLP_plain_regress(h_size=args.feature_out, embedding_size=args.embedding_size_output, y_size=10).to(args.device))
            ## Apply BCE and Sigmoid
            node_layers.append(MLP_plain_regress(h_size=args.feature_out, embedding_size=args.embedding_size_output, y_size=2).to(args.device))
            feature_extractor = MLP_plain_regress(h_size=args.hidden_size_rnn, embedding_size=args.feature_embedding, y_size=args.feature_out).to(args.device)
        elif 'GraphRNN_RNN' in args.note:
            rnn = GRU_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                            hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                            has_output=True, output_size=args.hidden_size_rnn_output,device=args.device).to(args.device)
            output =  MLP_plain(h_size=args.hidden_size_rnn_output, embedding_size=args.embedding_size_output, y_size=NUM_POS_EDGES).to(args.device)
   
            # output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
            #                    hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
            #                    has_output=True, output_size=NUM_POS_EDGES).to(args.device)
            # node_layer =  GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
            #                    hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
            #                    has_output=True, output_size=NUM_POS_ATOMS).to(args.device)
        ### start training
        train(args, dataset_loader, val_dataset_loader, rnn, output, node_layers=node_layers, feature_extractor=feature_extractor)
    sweep_id = wandb.sweep(sweep=sweep_config, project="GraphRNN-v2")
    wandb.agent(sweep_id=sweep_id, function=main)
    # main()
    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)

