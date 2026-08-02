import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

SAR_PROMPT = """You are drafting a Suspicious Activity Report (SAR) narrative section for a compliance analyst's review.
Given the following model explanation for a flagged transaction, write a 3-4 sentence factual narrative describing WHY this transaction was flagged, referencing the specific structural indicators. Do not invent facts not present in the data below. Use a formal, regulatory tone.

Flagged transaction ID: {node_id}
Model confidence (illicit probability): {confidence:.2%}
Number of influential connected transactions: {num_neighbors}
Top contributing features (indices): {top_features}

SAR Narrative:"""

def draft_sar(node_id, confidence, num_neighbors, top_features):
    prompt = SAR_PROMPT.format(
        node_id=node_id, confidence=confidence, num_neighbors=num_neighbors, top_features=top_features
    )
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    from explain_node import explain_flagged_node, data
    node_idx = (data.y == 1).nonzero(as_tuple=True)[0][0].item()
    result = explain_flagged_node(node_idx)
    sar = draft_sar(
        result["real_txid"], result["confidence"],
        len(set(result["important_neighbor_indices"])), result["top_feature_indices"]
    )
    print(sar)