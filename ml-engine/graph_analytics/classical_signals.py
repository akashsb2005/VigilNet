import networkx as nx
import community as community_louvain  # python-louvain

def compute_fan_ratios(G):
    """Structuring/smurfing signal: high fan-in or fan-out is suspicious."""
    ratios = {}
    for node in G.nodes():
        fan_in = G.in_degree(node)
        fan_out = G.out_degree(node)
        ratios[node] = {
            "fan_in": fan_in,
            "fan_out": fan_out,
            "pass_through_ratio": fan_out / (fan_in + 1)
        }
    return ratios

def detect_cycles(G, max_length=6):
    """Circular laundering chains — money returning to origin."""
    try:
        cycles = [c for c in nx.simple_cycles(G, length_bound=max_length)]
    except TypeError:
        cycles = [c for c in nx.simple_cycles(G) if len(c) <= max_length]
    return cycles

def detect_communities(G):
    """Louvain community detection — surfaces coordinated fraud rings."""
    undirected = G.to_undirected()
    partition = community_louvain.best_partition(undirected)
    return partition