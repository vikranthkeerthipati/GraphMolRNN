import pandas as pd
from train import *
from rdkit import Chem
import pandas as pd
import wandb
zinc_df = pd.read_csv('./SMILES_Keyboard.csv')
zinc_df = zinc_df.loc[zinc_df['SMILES'].str.len()<50][['SMILES', 'NAME']].reset_index(drop=True)
em = nx.algorithms.isomorphism.numerical_edge_match('bond_type',0)
for smStr in zinc_df["SMILES"]:
    mol = Chem.MolFromSmiles(smStr)
    G = mol_to_nx(mol)
    print(Chem.MolToSmiles(mol))
    print(Chem.MolToSmiles(nx_to_mol(G)))
    adj_copy = nx.attr_matrix(G, edge_attr="bond_type")[0]
    node_copy = np.array([np.array(list(G.nodes[n].values())) for n in range(G.number_of_nodes())])    
    x_idx = np.random.permutation(adj_copy.shape[0])
    adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
    node_copy = node_copy[x_idx]
    adj_copy_matrix = np.asmatrix(adj_copy)
    G_t = nx.from_numpy_matrix(adj_copy_matrix)
    start_idx = np.random.randint(adj_copy.shape[0])
    x_idx = np.array(bfs_seq(G_t, start_idx))
    adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
    node_copy = node_copy[x_idx]
    G_prime = get_mol_nodes(adj_copy,node_copy)
    print(Chem.MolToSmiles(nx_to_mol(G_prime)))
    adj_encoded, node_encoded = encode_adj(adj_copy.copy(), node_copy.copy(), max_prev_node=41)
    adj_decoded, node_decoded = decode_adj(adj_encoded, node_encoded)
    G_final = get_mol_nodes(adj_decoded,node_decoded)
    print(Chem.MolToSmiles(nx_to_mol(G_final)))

