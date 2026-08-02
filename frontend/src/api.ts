import axios from "axios";

const api = axios.create({ baseURL: "http://localhost:8000" });

export interface Alert {
  id: number;
  node_id: string;
  confidence: number;
  sar_narrative: string | null;
  status: string;
  created_at: string;
}

export interface SampleTx {
  txid: number;
  confidence: number;
}

export interface PaysimAccount {
  account: string;
  confidence: number;
}

export interface NewTransactionInput {
  amount: number;
  fan_in: number;
  fan_out: number;
  community_id?: number;
}

export const fetchAlerts = async (): Promise<Alert[]> => {
  const res = await api.get("/score/alerts");
  return res.data;
};

export const fetchSampleTransactions = async (): Promise<SampleTx[]> => {
  const res = await api.get("/score/sample");
  return res.data;
};

export const searchTransactions = async (query: string): Promise<SampleTx[]> => {
  const res = await api.get(`/score/search?q=${query}`);
  return res.data;
};

export const scoreTransaction = async (txid: number) => {
  const res = await api.post("/score/", { txid });
  return res.data;
};

export const fetchPaysimSample = async (): Promise<PaysimAccount[]> => {
  const res = await api.get("/score/paysim/sample");
  return res.data;
};

export const searchPaysimAccounts = async (query: string): Promise<PaysimAccount[]> => {
  const res = await api.get(`/score/paysim/search?q=${query}`);
  return res.data;
};

export const evaluateNewTransaction = async (input: NewTransactionInput) => {
  const res = await api.post("/score/evaluate", input);
  return res.data;
};

export default api;