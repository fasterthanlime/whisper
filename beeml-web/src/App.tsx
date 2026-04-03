import "./index.css";
import { useState } from "react";
import { RetrievalPrototypeLab } from "./components/RetrievalPrototypeLab";
import { TranscribeDemoPanel } from "./components/TranscribeDemoPanel";

export default function App() {
  const [tab, setTab] = useState<"transcribe" | "retrieval">("retrieval");
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:9944");

  return (
    <div className="app-shell">
      <header className="app-header">
        <strong>beeml-web</strong>
        <span className="subtitle">
          {tab === "transcribe" ? "Transcribe demo" : "Retrieval prototype lab"}
        </span>
        <div className="tab-row">
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
        {tab === "transcribe" ? (
          <TranscribeDemoPanel wsUrl={wsUrl} setWsUrl={setWsUrl} />
        ) : (
          <RetrievalPrototypeLab wsUrl={wsUrl} setWsUrl={setWsUrl} />
        )}
      </main>
    </div>
  );
}
