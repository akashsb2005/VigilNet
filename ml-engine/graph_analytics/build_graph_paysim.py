import pandas as pd
import networkx as nx

def build_paysim_graph(csv_path, fraud_ratio_neg=5):
    df = pd.read_csv(csv_path)

    fraud = df[df["isFraud"] == 1]
    legit_sample = df[df["isFraud"] == 0].sample(
        n=min(len(fraud) * fraud_ratio_neg, len(df)), random_state=42
    )
    sample = pd.concat([fraud, legit_sample]).reset_index(drop=True)
    print(f"Sampled {len(sample)} transactions ({len(fraud)} fraud, {len(legit_sample)} legit)")

    G = nx.DiGraph()
    for _, row in sample.iterrows():
        orig, dest = row["nameOrig"], row["nameDest"]
        G.add_node(orig)
        G.add_node(dest)
        G.add_edge(
            orig, dest,
            amount=row["amount"], type=row["type"], step=row["step"],
            is_fraud=row["isFraud"],
        )

    # Node label: an account is "illicit" if it touches ANY fraud transaction
    fraud_accounts = set(fraud["nameOrig"]) | set(fraud["nameDest"])
    for n in G.nodes():
        G.nodes[n]["label"] = 1 if n in fraud_accounts else 0

    return G, sample

if __name__ == "__main__":
    G, sample = build_paysim_graph("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
    print(f"Graph — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")