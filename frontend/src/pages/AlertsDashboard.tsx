import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import NetworkBackground from "../components/NetworkBackground";
import {
  fetchAlerts,
  scoreTransaction,
  fetchSampleTransactions,
  searchTransactions,
  fetchPaysimSample,
  searchPaysimAccounts,
  evaluateNewTransaction,
} from "../api";
import type { Alert, SampleTx, PaysimAccount } from "../api";
import api from "../api";

type Role = "analyst" | "auditor" | "manager";

function riskLevel(confidence: number): { label: string; color: string } {
  if (confidence >= 0.7) return { label: "HIGH RISK", color: "var(--color-risk-high)" };
  if (confidence >= 0.4) return { label: "WATCH", color: "var(--color-risk-med)" };
  return { label: "CLEARED", color: "var(--color-risk-low)" };
}

function RiskMeter({ confidence }: { confidence: number }) {
  const segments = 10;
  const filled = Math.round(confidence * segments);
  const { color } = riskLevel(confidence);
  return (
    <div className="flex gap-[3px]">
      {Array.from({ length: segments }).map((_, i) => (
        <motion.div
          key={i}
          className="w-[5px] h-4 rounded-sm"
          initial={{ opacity: 0, scaleY: 0 }}
          animate={{
            opacity: 1,
            scaleY: 1,
            background: i < filled ? color : "var(--color-border)",
          }}
          transition={{ delay: i * 0.03, duration: 0.25 }}
        />
      ))}
    </div>
  );
}

export default function AlertsDashboard() {
  const [role, setRole] = useState<Role>("analyst");
  const [network, setNetwork] = useState<"bitcoin" | "paysim">("bitcoin");
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [samples, setSamples] = useState<SampleTx[]>([]);
  const [paysimSamples, setPaysimSamples] = useState<PaysimAccount[]>([]);
  const [input, setInput] = useState("");
  const [searchResults, setSearchResults] = useState<(SampleTx | PaysimAccount)[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  // New-transaction evaluator state
  const [showEvaluator, setShowEvaluator] = useState(false);
  const [evalFanIn, setEvalFanIn] = useState("1");
  const [evalFanOut, setEvalFanOut] = useState("1");
  const [evalAmount, setEvalAmount] = useState("1000");
  const [evalResult, setEvalResult] = useState<{ confidence: number; note?: string } | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);

  const loadAlerts = async () => {
    const data = await fetchAlerts();
    setAlerts(data);
  };

  useEffect(() => {
    loadAlerts();
    fetchSampleTransactions().then(setSamples);
    fetchPaysimSample().then(setPaysimSamples);
  }, []);

  useEffect(() => {
    setInput("");
    setSearchResults([]);
    setError(null);
  }, [network]);

  const handleSearchInput = async (value: string) => {
    setInput(value);
    if (value.length < 3) {
      setSearchResults([]);
      return;
    }
    const results =
      network === "bitcoin" ? await searchTransactions(value) : await searchPaysimAccounts(value);
    setSearchResults(results);
  };

  const handleScore = async (override?: string) => {
    const identifier = override ?? input;
    if (!identifier) return;
    setLoading(true);
    setError(null);
    try {
      if (network === "bitcoin") {
        await scoreTransaction(Number(identifier));
      } else {
        await api.post("/score/paysim/", { account: identifier });
      }
      await loadAlerts();
      setInput("");
      setSearchResults([]);
    } catch {
      setError(
        network === "bitcoin"
          ? `Transaction ${identifier} not found in Bitcoin dataset.`
          : `Account ${identifier} not found in PaySim dataset.`
      );
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async () => {
    setEvalLoading(true);
    setEvalResult(null);
    try {
      const result = await evaluateNewTransaction({
        amount: Number(evalAmount),
        fan_in: Number(evalFanIn),
        fan_out: Number(evalFanOut),
      });
      setEvalResult(result);
    } finally {
      setEvalLoading(false);
    }
  };

  const currentSamples = network === "bitcoin" ? samples : paysimSamples;

  const highRiskCount = alerts.filter((a) => a.confidence >= 0.7).length;
  const watchCount = alerts.filter((a) => a.confidence >= 0.4 && a.confidence < 0.7).length;

  return (
    <div className="min-h-screen bg-transparent text-[var(--color-text)]">
      {/* Header */}
      <header className="relative overflow-hidden border-b border-[var(--color-border)] px-8 py-5 flex items-center justify-between">
        <NetworkBackground />
        <div className="relative z-10 flex items-center gap-3">
          <span className="relative flex items-center justify-center w-2 h-2">
            <span className="absolute w-2 h-2 rounded-full bg-[var(--color-risk-low)] radar-ping" />
            <span className="relative w-2 h-2 rounded-full bg-[var(--color-risk-low)]" />
          </span>
          <h1 className="font-mono text-lg font-semibold tracking-tight">
            VIGIL<span className="text-[var(--color-cyan)]">NET</span>
          </h1>
          <span className="text-xs text-[var(--color-muted)] font-mono ml-2">
            AML NETWORK INTELLIGENCE
          </span>
        </div>
        <div className="relative z-10 flex items-center gap-4">
          <select
            className="bg-[var(--color-surface)] border border-[var(--color-border)] rounded px-2 py-1 text-xs font-mono text-[var(--color-text)] focus:outline-none focus:border-[var(--color-cyan)]"
            value={role}
            onChange={(e) => setRole(e.target.value as Role)}
          >
            <option value="analyst">ANALYST VIEW</option>
            <option value="auditor">AUDITOR VIEW</option>
            <option value="manager">MANAGER VIEW</option>
          </select>
          <motion.span
            key={alerts.length}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-xs font-mono text-[var(--color-muted)]"
          >
            {alerts.length} case{alerts.length !== 1 ? "s" : ""} on file
          </motion.span>
        </div>
      </header>

      {/* Manager summary strip */}
      {role === "manager" && (
        <div className="flex gap-6 px-8 py-4 bg-[var(--color-surface)]/60 border-b border-[var(--color-border)] font-mono text-xs">
          <div>
            <span className="text-[var(--color-muted)]">TOTAL CASES </span>
            <span className="text-[var(--color-text)] font-semibold">{alerts.length}</span>
          </div>
          <div>
            <span className="text-[var(--color-muted)]">HIGH RISK </span>
            <span style={{ color: "var(--color-risk-high)" }} className="font-semibold">{highRiskCount}</span>
          </div>
          <div>
            <span className="text-[var(--color-muted)]">WATCH </span>
            <span style={{ color: "var(--color-risk-med)" }} className="font-semibold">{watchCount}</span>
          </div>
          <div>
            <span className="text-[var(--color-muted)]">HIGH RISK RATE </span>
            <span className="text-[var(--color-text)] font-semibold">
              {alerts.length ? ((highRiskCount / alerts.length) * 100).toFixed(0) : 0}%
            </span>
          </div>
        </div>
      )}

      {/* Network tabs */}
      <div className="relative flex gap-2 px-8 pt-4 bg-[var(--color-surface)]/60 backdrop-blur-sm border-b border-[var(--color-border)]">
        {(["bitcoin", "paysim"] as const).map((n) => (
          <button
            key={n}
            className={`relative font-mono text-xs px-4 py-2 transition ${
              network === n ? "text-[var(--color-cyan)]" : "text-[var(--color-muted)] hover:text-[var(--color-text)]"
            }`}
            onClick={() => setNetwork(n)}
          >
            {n === "bitcoin" ? "BITCOIN NETWORK · 203K TX" : "MOBILE MONEY · 6.3M TX"}
            {network === n && (
              <motion.div
                layoutId="tab-underline"
                className="absolute left-0 right-0 -bottom-px h-[2px] bg-[var(--color-cyan)]"
              />
            )}
          </button>
        ))}
        {role === "analyst" && (
          <button
            className="ml-auto font-mono text-xs px-4 py-2 text-[var(--color-cyan)] hover:brightness-110 transition"
            onClick={() => setShowEvaluator((v) => !v)}
          >
            {showEvaluator ? "✕ CLOSE EVALUATOR" : "+ EVALUATE NEW TRANSACTION"}
          </button>
        )}
      </div>

      {/* New transaction evaluator */}
      <AnimatePresence>
        {showEvaluator && role === "analyst" && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden bg-[var(--color-surface)]/80 border-b border-[var(--color-border)]"
          >
            <div className="px-8 py-5 max-w-3xl">
              <label className="text-xs font-mono text-[var(--color-muted)] uppercase tracking-wider">
                Evaluate a brand-new transaction (never seen in training)
              </label>
              <div className="flex gap-3 mt-2 flex-wrap items-end">
                <div>
                  <div className="text-[10px] font-mono text-[var(--color-muted)] mb-1">AMOUNT</div>
                  <input
                    className="w-28 bg-[var(--color-ink)] border border-[var(--color-border)] rounded px-2 py-1.5 font-mono text-sm focus:outline-none focus:border-[var(--color-cyan)]"
                    value={evalAmount}
                    onChange={(e) => setEvalAmount(e.target.value)}
                  />
                </div>
                <div>
                  <div className="text-[10px] font-mono text-[var(--color-muted)] mb-1">FAN-IN</div>
                  <input
                    className="w-20 bg-[var(--color-ink)] border border-[var(--color-border)] rounded px-2 py-1.5 font-mono text-sm focus:outline-none focus:border-[var(--color-cyan)]"
                    value={evalFanIn}
                    onChange={(e) => setEvalFanIn(e.target.value)}
                  />
                </div>
                <div>
                  <div className="text-[10px] font-mono text-[var(--color-muted)] mb-1">FAN-OUT</div>
                  <input
                    className="w-20 bg-[var(--color-ink)] border border-[var(--color-border)] rounded px-2 py-1.5 font-mono text-sm focus:outline-none focus:border-[var(--color-cyan)]"
                    value={evalFanOut}
                    onChange={(e) => setEvalFanOut(e.target.value)}
                  />
                </div>
                <button
                  className="bg-[var(--color-cyan)] text-[var(--color-ink)] font-mono text-sm font-semibold px-4 py-2 rounded disabled:opacity-40 hover:brightness-110 transition"
                  onClick={handleEvaluate}
                  disabled={evalLoading}
                >
                  {evalLoading ? "Evaluating…" : "Evaluate"}
                </button>
              </div>

              {evalResult && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 border rounded p-3"
                  style={{ borderColor: riskLevel(evalResult.confidence).color }}
                >
                  <div className="font-mono text-sm">
                    Confidence:{" "}
                    <span style={{ color: riskLevel(evalResult.confidence).color }} className="font-semibold">
                      {(evalResult.confidence * 100).toFixed(1)}% — {riskLevel(evalResult.confidence).label}
                    </span>
                  </div>
                  {evalResult.note && (
                    <div className="text-xs text-[var(--color-muted)] font-mono mt-1">{evalResult.note}</div>
                  )}
                </motion.div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Scoring console — hidden for auditor (read-only role) */}
      {role !== "auditor" && (
        <div className="px-8 py-6 border-b border-[var(--color-border)] bg-[var(--color-surface)]/40 backdrop-blur-sm">
          <div className="max-w-3xl">
            {currentSamples.length > 0 && (
              <div className="mb-5">
                <label className="text-xs font-mono text-[var(--color-muted)] uppercase tracking-wider">
                  Flagged {network === "bitcoin" ? "transactions" : "accounts"} in dataset — click to score
                </label>
                <div className="flex flex-wrap gap-2 mt-2">
                  {currentSamples.map((s, i) => {
                    const id = "txid" in s ? String(s.txid) : s.account;
                    return (
                      <motion.button
                        key={id}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.015 }}
                        whileHover={{ scale: 1.04, borderColor: "var(--color-cyan)" }}
                        whileTap={{ scale: 0.97 }}
                        className="font-mono text-xs border border-[var(--color-border)] rounded px-3 py-1.5 transition disabled:opacity-40"
                        onClick={() => handleScore(id)}
                        disabled={loading}
                      >
                        {id} <span className="text-[var(--color-muted)]">({(s.confidence * 100).toFixed(0)}%)</span>
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            )}

            <label className="text-xs font-mono text-[var(--color-muted)] uppercase tracking-wider">
              Search any {network === "bitcoin" ? "transaction ID" : "account"} (min 3 characters)
            </label>
            <div className="flex gap-2 mt-2">
              <input
                className="flex-1 bg-[var(--color-ink)] border border-[var(--color-border)] rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-[var(--color-cyan)] focus:shadow-[0_0_0_3px_rgba(34,211,238,0.15)] transition"
                placeholder={network === "bitcoin" ? "e.g. 93993009" : "e.g. C123456789"}
                value={input}
                onChange={(e) => handleSearchInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleScore()}
              />
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                className="relative overflow-hidden bg-[var(--color-cyan)] text-[var(--color-ink)] font-mono text-sm font-semibold px-5 py-2 rounded disabled:opacity-40 transition"
                onClick={() => handleScore()}
                disabled={loading}
              >
                {loading && (
                  <motion.span
                    className="absolute inset-0 bg-white/30"
                    initial={{ x: "-100%" }}
                    animate={{ x: "100%" }}
                    transition={{ repeat: Infinity, duration: 0.8, ease: "linear" }}
                  />
                )}
                <span className="relative z-10">{loading ? "Tracing…" : "Score"}</span>
              </motion.button>
            </div>

            <AnimatePresence>
              {searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-2 border border-[var(--color-border)] rounded max-h-48 overflow-y-auto"
                >
                  {searchResults.map((r) => {
                    const id = "txid" in r ? String(r.txid) : r.account;
                    return (
                      <button
                        key={id}
                        className="w-full text-left font-mono text-xs px-3 py-2 hover:bg-[var(--color-surface-raised)] transition flex justify-between"
                        onClick={() => handleScore(id)}
                      >
                        <span>{id}</span>
                        <span className="text-[var(--color-muted)]">{(r.confidence * 100).toFixed(1)}%</span>
                      </button>
                    );
                  })}
                </motion.div>
              )}
            </AnimatePresence>

            <AnimatePresence>
              {error && (
                <motion.p
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-xs font-mono text-[var(--color-risk-high)] mt-2"
                >
                  {error}
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Case queue */}
      <div className="px-8 py-6">
        {alerts.length === 0 ? (
          <p className="text-sm text-[var(--color-muted)] font-mono">
            No cases yet{role !== "auditor" ? " — score a transaction above to begin." : "."}
          </p>
        ) : (
          <div className="flex flex-col gap-2 max-w-4xl">
            <AnimatePresence>
              {alerts.map((a, i) => {
                const risk = riskLevel(a.confidence);
                const isOpen = expandedId === a.id;
                return (
                  <motion.div
                    key={a.id}
                    layout
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.02 }}
                    className="rounded-lg border overflow-hidden"
                    style={{
                      borderColor: isOpen ? risk.color : "var(--color-border)",
                      background: "var(--color-surface)",
                    }}
                  >
                    <button
                      className="w-full flex items-center gap-6 px-5 py-4 text-left hover:bg-[var(--color-surface-raised)] transition"
                      onClick={() => setExpandedId(isOpen ? null : a.id)}
                    >
                      <motion.div
                        className="w-1.5 self-stretch rounded-full"
                        animate={{ background: risk.color, boxShadow: `0 0 8px ${risk.color}66` }}
                      />
                      <div className="font-mono text-sm w-40 truncate">{a.node_id}</div>
                      <RiskMeter confidence={a.confidence} />
                      <div className="font-mono text-xs w-20" style={{ color: risk.color }}>
                        {risk.label}
                      </div>
                      <div className="font-mono text-sm ml-auto">{(a.confidence * 100).toFixed(1)}%</div>
                      {role === "auditor" ? (
                        <div className="text-xs text-[var(--color-muted)] font-mono">
                          {new Date(a.created_at).toLocaleDateString()}
                        </div>
                      ) : (
                        <div className="text-xs text-[var(--color-muted)] font-mono uppercase">{a.status}</div>
                      )}
                    </button>

                    <AnimatePresence>
                      {isOpen && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="px-5 pb-5 pt-1 border-t border-[var(--color-border)] overflow-hidden"
                        >
                          <div className="text-[10px] font-mono text-[var(--color-muted)] uppercase tracking-widest mb-2">
                            SAR narrative — draft, case #{a.node_id}
                          </div>
                          <p className="text-sm leading-relaxed text-[var(--color-text)]">
                            {a.sar_narrative || "No narrative generated (below alert threshold)."}
                          </p>
                          <div className="text-[10px] font-mono text-[var(--color-muted)] mt-3">
                            Filed {new Date(a.created_at).toLocaleString()}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  );
}