import "./index.css";
import { useEffect, useState } from "react";
import { JudgeEvalPanel } from "./components/JudgeEvalPanel";
import { JudgeRapidFirePanel } from "./components/JudgeRapidFirePanel";
import { RetrievalPrototypeLab } from "./components/RetrievalPrototypeLab";
import { TranscribeDemoPanel } from "./components/TranscribeDemoPanel";
import { OfflineJudgeEvalPanel } from "./components/OfflineJudgeEvalPanel";
import { CorrectionReviewPanel } from "./components/CorrectionReviewPanel";
import { PhoneticComparisonPanel } from "./components/PhoneticComparisonPanel";
import { CorpusCapturePanel } from "./components/CorpusCapturePanel";
import { CorpusAlignmentEvalPanel } from "./components/CorpusAlignmentEvalPanel";

const WS_URL = "ws://127.0.0.1:9944";
const TAB_ROUTES = {
  "correction-ui": "/correction-ui",
  "rapid-fire": "/rapid-fire",
  "judge-eval": "/judge-eval",
  "offline-eval": "/offline-eval",
  retrieval: "/retrieval",
  phonetics: "/phonetics",
  transcribe: "/transcribe",
  corpus: "/corpus",
  "corpus-eval": "/corpus-eval",
} as const;
type TabKey = keyof typeof TAB_ROUTES;
const ROUTE_TABS = Object.fromEntries(
  Object.entries(TAB_ROUTES).map(([tab, route]) => [route, tab]),
) as Record<(typeof TAB_ROUTES)[TabKey], TabKey>;

function normalizePath(pathname: string): string {
  if (pathname === "/") {
    return "/corpus-eval";
  }
  const trimmed = pathname.endsWith("/") && pathname !== "/" ? pathname.slice(0, -1) : pathname;
  return ROUTE_TABS[trimmed as keyof typeof ROUTE_TABS] ? trimmed : "/corpus-eval";
}

function tabForPath(pathname: string): TabKey {
  return ROUTE_TABS[normalizePath(pathname) as keyof typeof ROUTE_TABS] ?? "corpus-eval";
}

function navigateTo(pathname: string, replace = false) {
  const method = replace ? "replaceState" : "pushState";
  window.history[method](null, "", pathname);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export default function App() {
  const [tab, setTab] = useState<TabKey>(() =>
    typeof window === "undefined" ? "corpus-eval" : tabForPath(window.location.pathname),
  );

  useEffect(() => {
    const applyLocation = () => {
      const normalizedPath = normalizePath(window.location.pathname);
      if (window.location.pathname !== normalizedPath) {
        navigateTo(normalizedPath, true);
        return;
      }
      setTab(tabForPath(normalizedPath));
    };

    applyLocation();
    window.addEventListener("popstate", applyLocation);
    return () => window.removeEventListener("popstate", applyLocation);
  }, []);

  const selectTab = (nextTab: TabKey) => {
    const route = TAB_ROUTES[nextTab];
    if (window.location.pathname !== route) {
      navigateTo(route);
    } else {
      setTab(nextTab);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <strong>beeml</strong>
        <div className="tab-row" role="tablist">
          <button
            role="tab"
            aria-selected={tab === "correction-ui"}
            className={tab === "correction-ui" ? "primary" : ""}
            onClick={() => selectTab("correction-ui")}
          >
            Correction UI
          </button>
          <button
            role="tab"
            aria-selected={tab === "rapid-fire"}
            className={tab === "rapid-fire" ? "primary" : ""}
            onClick={() => selectTab("rapid-fire")}
          >
            Rapid Fire
          </button>
          <button
            role="tab"
            aria-selected={tab === "judge-eval"}
            className={tab === "judge-eval" ? "primary" : ""}
            onClick={() => selectTab("judge-eval")}
          >
            Judge Eval
          </button>
          <button
            role="tab"
            aria-selected={tab === "offline-eval"}
            className={tab === "offline-eval" ? "primary" : ""}
            onClick={() => selectTab("offline-eval")}
          >
            Offline Eval
          </button>
          <button
            role="tab"
            aria-selected={tab === "retrieval"}
            className={tab === "retrieval" ? "primary" : ""}
            onClick={() => selectTab("retrieval")}
          >
            Retrieval Lab
          </button>
          <button
            role="tab"
            aria-selected={tab === "phonetics"}
            className={tab === "phonetics" ? "primary" : ""}
            onClick={() => selectTab("phonetics")}
          >
            Phonetics
          </button>
          <button
            role="tab"
            aria-selected={tab === "transcribe"}
            className={tab === "transcribe" ? "primary" : ""}
            onClick={() => selectTab("transcribe")}
          >
            Transcribe
          </button>
          <button
            role="tab"
            aria-selected={tab === "corpus"}
            className={tab === "corpus" ? "primary" : ""}
            onClick={() => selectTab("corpus")}
          >
            Corpus
          </button>
          <button
            role="tab"
            aria-selected={tab === "corpus-eval"}
            className={tab === "corpus-eval" ? "primary" : ""}
            onClick={() => selectTab("corpus-eval")}
          >
            Corpus Eval
          </button>
        </div>
      </header>
      <main className="app-main">
        <div className="app-page">
          {tab === "transcribe" ? (
            <TranscribeDemoPanel wsUrl={WS_URL} />
          ) : tab === "corpus" ? (
            <CorpusCapturePanel wsUrl={WS_URL} />
          ) : tab === "corpus-eval" ? (
            <CorpusAlignmentEvalPanel wsUrl={WS_URL} />
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
