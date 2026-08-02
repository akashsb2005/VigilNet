<div align="center">
# 🛡️ VigilNet
### AML & Financial Crime Network Intelligence Platform
 
**Graph Neural Networks × Explainable AI × LLM-Powered Compliance Automation**
 
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
 
*Detecting money laundering networks the way modern financial institutions actually do — through graph structure, not isolated transactions.*
 
[Live Demo](#) · [Architecture](#architecture) · [Results](#results--benchmarks) · [API Docs](#api-reference) · [Setup](#getting-started)
 
</div>
---
 
## Table of Contents
 
- [Why This Exists](#why-this-exists)
- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Results & Benchmarks](#results--benchmarks)
- [Design Decisions & Research Gaps Addressed](#design-decisions--research-gaps-addressed)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [About the Developer](#about-the-developer)
- [License](#license)
---
 
## Why This Exists
 
Most academic fraud-detection projects stop at reporting accuracy on a single dataset — a number that means almost nothing when illicit transactions make up under 2% of the data. A model that predicts "legitimate" for everything scores 98%+ accuracy and catches zero fraud.
 
Real financial institutions have moved past single-transaction rule engines because organized financial crime *is* a network phenomenon: money launderers structure transactions across many accounts specifically to stay under detection thresholds. Published industry research (HSBC's KDD 2025 work on graph-based AML) shows heterogeneous GNNs cutting false positives by ~30% while catching materially more previously-unknown laundering networks than legacy systems.
 
**VigilNet was built to reflect that reality** — with the statistical rigor (temporal validation, honest baseline comparisons, precision-recall as the real metric) that most student projects skip, and with the explainability layer that turns a bare probability score into an actual regulator-usable document.
 
## What It Does
 
1. **Ingests a transaction network** — either a Bitcoin transaction graph or a mobile-money account graph
2. **Scores every node** with a Graph Attention Network (GAT), trained with a strict **temporal walk-forward split** — never random, since fraud patterns evolve over time and random splits leak future information into training
3. **Explains every flagged case** using GNNExplainer, identifying exactly which connected transactions and which features drove the model's decision
4. **Auto-drafts a Suspicious Activity Report (SAR) narrative** via an LLM (Groq / Llama 3.3), using the structured explanation as grounding — the actual document format compliance analysts file with regulators
5. **Persists and serves cases** through a FastAPI + PostgreSQL backend
6. **Presents a live triage dashboard** — React + TypeScript, with role-based views for analysts, auditors, and managers, real-time transaction evaluation, and animated network visualization
---
 
## Architecture
 
```
┌───────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                │
│   Elliptic++ (203K Bitcoin tx)      PaySim (6.3M mobile-money tx)  │
└──────────────────────┬────────────────────────┬───────────────────┘
                        │                        │
                        ▼                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                    CLASSICAL GRAPH ANALYTICS                       │
│     Fan-in / fan-out ratios · Louvain community detection          │
│     Cycle detection (circular laundering chains)                   │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│              GRAPH ATTENTION NETWORK (PyTorch Geometric)            │
│     Temporal walk-forward train/val/test split                     │
│     Class-weighted loss + gradient clipping                        │
│     Benchmarked against XGBoost baseline on identical splits        │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                        EXPLAINABILITY LAYER                        │
│     GNNExplainer → driving subgraph + feature importance            │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    LLM COMPLIANCE AUTOMATION                       │
│     Groq (Llama 3.3-70B) + LangChain → SAR narrative drafting       │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                   FASTAPI + POSTGRESQL BACKEND                     │
│     REST API · Case persistence · Alembic migrations                │
└──────────────────────────────┬────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│              REACT + TYPESCRIPT TRIAGE DASHBOARD                   │
│     Live case queue · Role-based views · Real-time evaluation       │
│     Framer Motion network visualization                             │
└───────────────────────────────────────────────────────────────────┘
```
 
---
 
## Results & Benchmarks
 
Validated across two structurally different transaction networks, with an honest baseline comparison rather than reporting the GNN's number in isolation:
 
| Dataset | Network Type | Nodes (analyzed) | Test PR-AUC (GAT) | Test PR-AUC (XGBoost baseline) | Key Finding |
|---|---|---|---|---|---|
| **Elliptic++** | Bitcoin transaction graph | 203,769 | **0.30** | comparable | Severe temporal distribution shift between train and test periods — a measured, real finding, not a training failure |
| **PaySim** | Mobile-money account graph | ~49,000 (stratified sample) | **0.95** | lower | Smooth convergence, no oscillation; cleaner structuring-based fraud signal |
 
**Why report a "failure" case at all?** Because it's true, and because the gap between 0.30 and 0.95 using the *identical* architecture is the actual research finding: fraud-detection performance is deeply dataset- and typology-dependent. A portfolio that only shows the good number teaches nothing; showing both proves the pipeline generalizes architecturally while being transparent about where the hard problem actually lives.
 
### Classical graph signal validation
 
Before any neural model was trained, the classical features were validated against ground truth:
- Raw pass-through ratio showed **no separation** between illicit and licit nodes (0.719 vs 0.721 mean) — an honest negative result
- Louvain community concentration showed **real signal** — up to 33% illicit concentration in specific communities vs a ~2% base rate, confirming coordinated fraud rings cluster detectably
---
 
## Design Decisions & Research Gaps Addressed
 
| Gap in typical projects | VigilNet's approach |
|---|---|
| Reports accuracy on ~2% imbalanced data | Precision-recall / PR-AUC as the primary metric throughout |
| Random train/test split leaks future information | Strict temporal walk-forward splits, timestep-aware |
| Assumes graph structure always helps | XGBoost baseline run on identical splits to test this honestly |
| Plain GCN mean-aggregation gets fooled by camouflage | GAT (attention-based) aggregation, down-weights misleading neighbors |
| Node-classification only, ignores network topology | Classical graph analytics (fan-in/out, Louvain, cycles) run alongside the GNN |
| Stops at a probability score | GNNExplainer → LLM pipeline produces an actual SAR narrative |
| Single dataset, unverified generalization | Same architecture validated on two structurally different networks |
| Notebook-only, never deployed | Full FastAPI + PostgreSQL + React service, containerized, deployed |
 
---
 
## Tech Stack
 
**Machine Learning & Graph Analytics**
`PyTorch` · `PyTorch Geometric` · `NetworkX` · `python-louvain` · `scikit-learn` · `XGBoost`
 
**Explainability & LLM**
`GNNExplainer` · `Groq API (Llama 3.3-70B)` · `LangChain`
 
**Backend**
`FastAPI` · `SQLAlchemy` · `Alembic` · `PostgreSQL` · `Pydantic`
 
**Frontend**
`React 18` · `TypeScript` · `Vite` · `Tailwind CSS v4` · `Framer Motion` · `Axios`
 
**Infrastructure**
`Docker` · `Git LFS` · `Render`
 
---
 
## Project Structure
 
```
VigilNet/
├── ml-engine/
│   ├── data/                    # Raw + processed datasets (Git LFS)
│   ├── graph_analytics/         # Classical algorithms (fan-in/out, Louvain, cycles)
│   ├── models/                  # GAT architecture, training, temporal splits
│   └── explain/                 # GNNExplainer + LLM SAR drafting
├── backend/
│   ├── app/
│   │   ├── models/              # SQLAlchemy: Alert
│   │   ├── routers/             # /score, /score/paysim endpoints
│   │   ├── services/            # ML inference bridge, SAR generation
│   │   └── core/                # DB session, config
│   ├── tests/                   # pytest suite
│   └── alembic/                 # Migrations
├── frontend/
│   └── src/
│       ├── pages/                # AlertsDashboard
│       ├── components/           # NetworkBackground (canvas animation)
│       └── api.ts                # Typed API client
└── infra/
    └── docker-compose.yml
```
 
---
 
## Getting Started
 
### Prerequisites
- Python 3.11+
- Node 20+
- PostgreSQL 16+
- A free [Groq API key](https://console.groq.com)
### Setup
 
```powershell
# Database
createdb vigilnet
 
# ML engine
cd ml-engine
python -m venv venv && .\venv\Scripts\activate
pip install torch torch_geometric networkx python-louvain scikit-learn xgboost pandas numpy langchain-groq python-dotenv
 
# Datasets — see "Datasets" section below for sources
python graph_analytics/build_graph.py
python models/build_dataset.py
python models/train_gnn.py
python models/build_dataset_paysim.py
python models/train_gnn_paysim.py
 
# Backend
cd ../backend
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
 
# Frontend (new terminal)
cd ../frontend
npm install
npm run dev
```
 
### Datasets
 
Raw data and trained model artifacts are tracked via Git LFS rather than committed as plain files, given their size. Sources:
- **Elliptic++:** [git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus)
- **PaySim:** [Kaggle — ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
---
 
## API Reference
 
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/score/sample` | Top flagged Bitcoin transactions |
| `GET` | `/score/search?q=` | Search Bitcoin transaction IDs |
| `POST` | `/score/` | Score a Bitcoin transaction by ID, generate SAR if flagged |
| `POST` | `/score/evaluate` | Score a brand-new transaction using engineered features only |
| `GET` | `/score/paysim/sample` | Top flagged PaySim accounts |
| `GET` | `/score/paysim/search?q=` | Search PaySim accounts |
| `POST` | `/score/paysim/` | Score a PaySim account, generate SAR if flagged |
| `GET` | `/score/alerts` | List persisted cases |
 
Full interactive documentation available at `/docs` (Swagger UI) when running locally.
 
---
 
## Testing
 
```powershell
cd backend
python -m pytest tests/
```
 
Covers health check integrity, invalid-ID error handling, and response schema validation for the sample endpoint.
 
---
 
## Deployment
 
Containerized with Docker for both backend and frontend, deployed on Render with a managed PostgreSQL instance. See `infra/docker-compose.yml` for local multi-container orchestration.
 
**Known cold-start behavior:** the free-tier backend spins down after inactivity; first request after idle may take 30–60 seconds to respond while the container restarts.
 
---
 
## Known Limitations & Roadmap
 
Stated plainly, because interviewers respect this more than overselling:
 
- PaySim's account-to-index mapping rebuilds from the raw CSV on every backend startup rather than being cached — fine for a demo, not production-grade
- A brand-new transaction scored via `/score/evaluate` has no real graph neighborhood yet, since GNNs rely heavily on connection structure — the score reflects engineered features only
- True production scale requires distributed graph sampling (e.g., GraphSAGE-style neighbor sampling) rather than loading the full graph into memory
- The Bitcoin model's low test PR-AUC under temporal shift is a genuine open research problem — likely requiring online/continual learning rather than a static split to meaningfully improve
- Model artifacts are versioned via Git LFS rather than a proper model registry — a real next step for production maturity
**Planned next:**
- [ ] Incremental graph construction for real-time new-transaction scoring with actual neighborhood context
- [ ] Model registry integration (MLflow or similar)
- [ ] Distributed neighbor sampling for full-scale training
---
 
## About the Developer
 
**Akash S B** is a 3rd-year B.Tech student in Computer Science (AI & ML) at PES University, RR Campus, graduating in 2028. He currently interns as a **DevSpace Tech Intern at BSERC (ISRO)** and serves as an **R&D Member at PI Labs, PESU**, and holds both the Reliance Foundation and FFE scholarships.
 
He's an active competitive programmer with 200+ LeetCode problems and 170+ Codeforces problems solved (rating ~1120), and builds portfolio-grade, production-oriented projects spanning AI/ML, cybersecurity, and full-stack engineering — including SentinelAI (a document/identity fraud detection platform), NeuroPulse (multimodal fatigue detection), and Industrial Knowledge Brain (a multi-agent RAG system built for the ET AI Hackathon).
 
VigilNet reflects his particular interest in combining rigorous ML methodology with real deployable systems — not just notebooks, but services with explainability, honest benchmarking, and genuine production considerations built in from the start.
 
**Connect:** [GitHub](https://github.com/akashsb2005)
 
---
 
## License
 
MIT License — see [LICENSE](LICENSE) for details.
 
</div>
 
