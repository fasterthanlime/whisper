import "./index.css";
import { useState } from "react";
import { JudgeEvalPanel } from "./components/JudgeEvalPanel";
import { JudgeRapidFirePanel } from "./components/JudgeRapidFirePanel";
import { RetrievalPrototypeLab } from "./components/RetrievalPrototypeLab";
import { TranscribeDemoPanel } from "./components/TranscribeDemoPanel";

const WS_URL = "ws://127.0.0.1:9944";

export default function App() {
  const [tab, setTab] = useState<
    "transcribe" | "retrieval" | "rapid-fire" | "judge-eval"
  >("rapid-fire");

  const subtitle =
    tab === "transcribe"
      ? "Transcribe demo"
      : tab === "retrieval"
        ? "Retrieval prototype lab"
        : tab === "rapid-fire"
          ? "Judge rapid fire"
          : "Judge eval";

  return (
    <div className="app-shell">
      <header className="app-header">
        <strong>beeml-web</strong>
        <span className="subtitle">{subtitle}</span>
        <div className="tab-row">
          <button
            className={tab === "rapid-fire" ? "primary" : ""}
            onClick={() => setTab("rapid-fire")}
          >
            Rapid Fire
          </button>
          <button
            className={tab === "judge-eval" ? "primary" : ""}
            onClick={() => setTab("judge-eval")}
          >
            Judge Eval
          </button>
          <button
            className={tab === "retrieval" ? "primary" : ""}
            onClick={() => setTab("retrieval")}
          >
            Retrieval Lab
          </button>
          <button
            className={tab === "transcribe" ? "primary" : ""}
            onClick={() => setTab("transcribe")}
          >
            Transcribe
          </button>
        </div>
      </header>
      <main className="app-main">
        <div className="app-page">
          {tab === "transcribe" ? (
            <TranscribeDemoPanel wsUrl={WS_URL} />
          ) : tab === "rapid-fire" ? (
            <JudgeRapidFirePanel wsUrl={WS_URL} />
          ) : tab === "judge-eval" ? (
            <JudgeEvalPanel wsUrl={WS_URL} />
          ) : (
            <RetrievalPrototypeLab wsUrl={WS_URL} />
          )}
        </div>
      </main>
    </div>
  );
}
