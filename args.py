
### program configuration
import torch
class Args(object):
    def __init__(self, my_dict=None):
        if my_dict:
            for key in my_dict:
                setattr(self, key, my_dict[key])
        ### if clean tensorboard
        if not hasattr(self, "clean_tensorboard"):
            self.clean_tensorboard = False
        ### Which CUDA GPU device is used for training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ### Which GraphRNN model variant is used.
        # The simple version of Graph RNN
        self.note = 'GraphRNN_MLP'
        # The dependent Bernoulli sequence version of GraphRNN
        # self.note = 'GraphRNN_RNN'

        ## for comparison, removing the BFS compoenent
        # self.note = 'GraphRNN_MLP_nobfs'
        # self.note = 'GraphRNN_RNN_nobfs'

        ### Which dataset is used to train the model
        # self.graph_type = 'DD'
        # self.graph_type = 'caveman'
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'caveman_small_single'
        # self.graph_type = 'community4'
        # self.graph_type = 'grid'
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer'
        # self.graph_type = 'citeseer_small'
        self.graph_type = 'zinc'
        # self.graph_type = 'barabasi_noise'
        # self.noise = 10
        #
        # if self.graph_type == 'barabasi_noise':
        #     self.graph_type = self.graph_type+str(self.noise)

        # if none, then auto calculate
        self.max_num_node = None # max number of nodes in a graph
        self.max_prev_node = None # max previous node that looks back

        ### network config
        ## GraphRNN
        # if 'small' in self.graph_type:
        #     self.parameter_shrink = 2
        # else:
        if not hasattr(self, "parameter_shrink"):
            self.parameter_shrink = 1
        if not hasattr(self, "hidden_size_rnn"):
            self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        if not hasattr(self, "hidden_size_rnn_output"):
            self.hidden_size_rnn_output = 16 # hidden size for output RNN
        if not hasattr(self, "embedding_size_rnn"):
            self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        if not hasattr(self, "embedding_size_rnn_output"):
            self.embedding_size_rnn_output = 8 # the embedding size for output rnn
        if not hasattr(self, "embedding_size_output"):
            self.embedding_size_output = int(64/self.parameter_shrink) # the embedding size for output (VAE/MLP)
        if not hasattr(self, "embedding_size_node"):
            self.embedding_size_node = 64 # the embedding size for output (VAE/MLP)
        if not hasattr(self, "batch_size"):
            self.batch_size = 128 # normal: 32, and the rest should be changed accordingly
        if not hasattr(self, "test_batch_size"):
            self.test_batch_size = 128
        if not hasattr(self, "test_total_size"):
            self.test_total_size = 1000
        if not hasattr(self, "num_layers"):
            self.num_layers = 4

        ### training config
        if not hasattr(self, "num_workers"):
            self.num_workers = 4 # num workers to load data, default 4
        if not hasattr(self, "batch_ratio"):
            self.batch_ratio = 128 # how many batches of samples per epoch, default 32, e.g., 1 epoch = 32 batches
        if not hasattr(self, "epochs"):
            self.epochs = 50 # now one epoch means self.batch_ratio x batch_size
        if not hasattr(self, "epochs_test_start"):
            self.epochs_test_start = 100
        if not hasattr(self, "epochs_test"):
            self.epochs_test = 100
        if not hasattr(self, "epochs_log"):
            self.epochs_log = 100
        if not hasattr(self, "epochs_save"):
            self.epochs_save = 100

        if not hasattr(self, "lr"):
            self.lr = 0.000003
        if not hasattr(self, "output_lr"):
            self.output_lr = self.lr
        if not hasattr(self, "node_lr"):
            self.node_lr = 0.000003
        if not hasattr(self, "atomic_lr"):
            self.atomic_lr = self.lr
        if not hasattr(self, "formal_lr"):
            self.formal_lr = self.lr
        if not hasattr(self, "chiral_lr"):
            self.chiral_lr = self.lr
        if not hasattr(self, "hybr_lr"):
            self.hybr_lr = self.lr
        if not hasattr(self, "numh_lr"):
            self.numh_lr = self.lr
        if not hasattr(self, "aromatic_lr"):
            self.aromatic_lr = self.node_lr
        
        if not hasattr(self, "milestones"):
            self.milestones = [400, 1000]
        if not hasattr(self, "lr_rate"):
            self.lr_rate = 0.3
        if not hasattr(self, "feature_embedding"):
            self.feature_embedding = 64
        if not hasattr(self, "feature_out"):
            self.feature_out = self.feature_embedding // 2
        if not hasattr(self, "atomic_loss_factor"):
            self.atomic_loss_factor = 1

        self.sample_time = 2 # sample time in each time step, when validating

        ### output config
        # self.dir_input = "/dfs/scratch0/jiaxuany0/"
        self.dir_input = "./"
        self.model_save_path = self.dir_input+'model_save/' # only for nll evaluation
        self.graph_save_path = self.dir_input+'graphs/'
        self.figure_save_path = self.dir_input+'figures/'
        self.timing_save_path = self.dir_input+'timing/'
        self.figure_prediction_save_path = self.dir_input+'figures_prediction/'
        self.nll_save_path = self.dir_input+'nll/'


        self.load = False # if load model, default lr is very low
        self.load_epoch = 3000
        self.save = True


        ### baseline config
        # self.generator_baseline = 'Gnp'
        self.generator_baseline = 'BA'

        # self.metric_baseline = 'general'
        # self.metric_baseline = 'degree'
        self.metric_baseline = 'clustering'


        ### filenames to save intemediate and final outputs
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_'
        self.fname_pred = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_pred_'
        self.fname_train = self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+ str(self.hidden_size_rnn)+'_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.hidden_size_rnn) + '_test_'
        self.fname_baseline = self.graph_save_path + self.graph_type + self.generator_baseline+'_'+self.metric_baseline

