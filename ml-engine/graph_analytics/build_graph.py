import pandas as pd
import networkx as nx

def build_elliptic_graph(features_path, edges_path, classes_path):
    # header=0 (default) because these files DO have header rows (txId, etc.)
    features = pd.read_csv(features_path, low_memory=False)
    edges = pd.read_csv(edges_path)
    classes = pd.read_csv(classes_path)

    G = nx.DiGraph()

    # Use .iloc for position-based access so we don't depend on exact column names
    for row in features.itertuples(index=False):
        node_id = int(row[0])
        timestep = int(row[1])
        feature_values = row[2:]
        G.add_node(node_id, timestep=timestep, features=feature_values)

    for row in edges.itertuples(index=False):
        G.add_edge(int(row[0]), int(row[1]))

    classes.columns = classes.columns.str.strip()
    label_map = dict(zip(classes.iloc[:, 0], classes.iloc[:, 1]))
    nx.set_node_attributes(G, label_map, name='label')

    return G

if __name__ == "__main__":
    G = build_elliptic_graph(
        "data/raw/elliptic/txs_features.csv",
        "data/raw/elliptic/txs_edgelist.csv",
        "data/raw/elliptic/txs_classes.csv",
    )
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")