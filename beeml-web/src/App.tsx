import "./index.css";
import { useState } from "react";
import { JudgeEvalPanel } from "./components/JudgeEvalPanel";
import { JudgeRapidFirePanel } from "./components/JudgeRapidFirePanel";
import { RetrievalPrototypeLab } from "./components/RetrievalPrototypeLab";
import { TranscribeDemoPanel } from "./components/TranscribeDemoPanel";
import { OfflineJudgeEvalPanel } from "./components/OfflineJudgeEvalPanel";
import { CorrectionReviewPanel } from "./components/CorrectionReviewPanel";
import { PhoneticComparisonPanel } from "./components/PhoneticComparisonPanel";
import { CorpusCapturePanel } from "./components/CorpusCapturePanel";

const WS_URL = "ws://127.0.0.1:9944";

export default function App() {
  const [tab, setTab] = useState<
    "transcribe" | "retrieval" | "rapid-fire" | "judge-eval" | "offline-eval" | "correction-ui" | "phonetics" | "corpus"
  >("correction-ui");

  return (
    <div className="app-shell">
      <header className="app-header">
        <strong>beeml</strong>
        <div className="tab-row" role="tablist">
          <button
            role="tab"
            aria-selected={tab === "correction-ui"}
            className={tab === "correction-ui" ? "primary" : ""}
            onClick={() => setTab("correction-ui")}
          >
            Correction UI
          </button>
          <button
            role="tab"
            aria-selected={tab === "rapid-fire"}
            className={tab === "rapid-fire" ? "primary" : ""}
            onClick={() => setTab("rapid-fire")}
          >
            Rapid Fire
          </button>
          <button
            role="tab"
            aria-selected={tab === "judge-eval"}
            className={tab === "judge-eval" ? "primary" : ""}
            onClick={() => setTab("judge-eval")}
          >
            Judge Eval
          </button>
          <button
            role="tab"
            aria-selected={tab === "offline-eval"}
            className={tab === "offline-eval" ? "primary" : ""}
            onClick={() => setTab("offline-eval")}
          >
            Offline Eval
          </button>
          <button
            role="tab"
            aria-selected={tab === "retrieval"}
            className={tab === "retrieval" ? "primary" : ""}
            onClick={() => setTab("retrieval")}
          >
            Retrieval Lab
          </button>
          <button
            role="tab"
            aria-selected={tab === "phonetics"}
            className={tab === "phonetics" ? "primary" : ""}
            onClick={() => setTab("phonetics")}
          >
            Phonetics
          </button>
          <button
            role="tab"
            aria-selected={tab === "transcribe"}
            className={tab === "transcribe" ? "primary" : ""}
            onClick={() => setTab("transcribe")}
          >
            Transcribe
          </button>
          <button
            role="tab"
            aria-selected={tab === "corpus"}
            className={tab === "corpus" ? "primary" : ""}
            onClick={() => setTab("corpus")}
          >
            Corpus
          </button>
        </div>
      </header>
      <main className="app-main">
        <div className="app-page">
          {tab === "transcribe" ? (
            <TranscribeDemoPanel wsUrl={WS_URL} />
          ) : tab === "corpus" ? (
            <CorpusCapturePanel wsUrl={WS_URL} />
          ) : tab === "correction-ui" ? (
            <CorrectionReviewPanel />
          ) : tab === "rapid-fire" ? (
            <JudgeRapidFirePanel wsUrl={WS_URL} />
          ) : tab === "judge-eval" ? (
            <JudgeEvalPanel wsUrl={WS_URL} />
          ) : tab === "offline-eval" ? (
            <OfflineJudgeEvalPanel wsUrl={WS_URL} />
          ) : tab === "phonetics" ? (
            <PhoneticComparisonPanel wsUrl={WS_URL} />
          ) : (
            <RetrievalPrototypeLab wsUrl={WS_URL} />
          )}
        </div>
      </main>
    </div>
  );
}
