import { useState } from "react";
import "./index.css";
import { LiveEvalPanel } from "./components/LiveEvalPanel";
import { HumanEvalPanel } from "./components/HumanEvalPanel";
import { RetrievalDebugPanel } from "./components/RetrievalDebugPanel";

type Tab = "live" | "human" | "retrieval";

export default function App() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          padding: "0.5rem 1rem",
          borderBottom: "1px solid var(--border)",
          background: "var(--bg-surface)",
        }}
      >
        <strong>beeml-web</strong>
        <nav style={{ display: "flex", gap: "0.25rem" }}>
          <button
            className={tab === "live" ? "primary" : ""}
            onClick={() => setTab("live")}
          >
            Live
          </button>
          <button
            className={tab === "human" ? "primary" : ""}
            onClick={() => setTab("human")}
          >
            Human Eval
          </button>
          <button
            className={tab === "retrieval" ? "primary" : ""}
            onClick={() => setTab("retrieval")}
          >
            Retrieval
          </button>
        </nav>
      </header>
      <main style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {tab === "live" ? (
          <LiveEvalPanel />
        ) : tab === "human" ? (
          <HumanEvalPanel />
        ) : (
          <RetrievalDebugPanel />
        )}
      </main>
    </div>
  );
}
