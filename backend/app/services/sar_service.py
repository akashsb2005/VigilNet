import sys, os
ML_ENGINE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml-engine')
sys.path.append(os.path.join(ML_ENGINE_PATH, 'explain'))

from dotenv import load_dotenv
load_dotenv(os.path.join(ML_ENGINE_PATH, '.env'))

from sar_writer import draft_sar

def generate_sar(txid, confidence, num_neighbors=5, top_features=None):
    return draft_sar(txid, confidence, num_neighbors, top_features or [])