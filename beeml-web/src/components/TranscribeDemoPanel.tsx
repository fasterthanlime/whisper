import { useCallback, useRef, useState } from "react";
import { channel } from "@bearcove/vox-core";
import { connectBeeMl } from "../beeml.generated";
import type {
  Update,
  AlignedWord,
  TranscribePhoneticTrace as RpcTranscribePhoneticTrace,
} from "../beeml.generated";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { EvalInspector } from "./EvalInspector";
import type { EvalInspectorData, PhoneticRescueTrace, TimedToken } from "../types";

function alignedWordsToTimedTokens(words: AlignedWord[]): TimedToken[] {
  return words.map((word) => ({
    w: word.word,
    s: word.start,
    e: word.end,
    meanLogprob: word.confidence.mean_lp,
    minLogprob: word.confidence.min_lp,
    meanMargin: word.confidence.mean_m,
    minMargin: word.confidence.min_m,
  }));
}

function toPhoneticTrace(trace: RpcTranscribePhoneticTrace): PhoneticRescueTrace {
  return {
    utteranceZipaRaw: trace.utterance_zipa_raw,
    utteranceZipaNormalized: trace.utterance_zipa_normalized,
    utteranceTranscriptNormalized: trace.utterance_transcript_normalized,
    utteranceSimilarity: trace.utterance_similarity,
    utteranceFeatureSimilarity: trace.utterance_feature_similarity,
    utteranceAlignment: trace.utterance_alignment.map((op) => ({
      kind: op.kind.tag,
      transcriptIndex: op.transcript_index,
      zipaIndex: op.zipa_index,
      transcriptToken: op.transcript_token,
      zipaToken: op.zipa_token,
      cost: op.cost,
    })),
    spans: trace.spans.map((span) => ({
      spanText: span.span_text,
      tokenStart: span.token_start,
      tokenEnd: span.token_end,
      startSec: span.start_sec,
      endSec: span.end_sec,
      zipaNormStart: span.zipa_norm_start,
      zipaNormEnd: span.zipa_norm_end,
      zipaRaw: span.zipa_raw,
      zipaNormalized: span.zipa_normalized,
      transcriptNormalized: span.transcript_normalized,
      transcriptPhoneCount: span.transcript_phone_count,
      chosenZipaPhoneCount: span.chosen_zipa_phone_count,
      transcriptSimilarity: span.transcript_similarity,
      transcriptFeatureSimilarity: span.transcript_feature_similarity,
      projectedAlignmentScore: span.projected_alignment_score,
      chosenAlignmentScore: span.chosen_alignment_score,
      secondBestAlignmentScore: span.second_best_alignment_score,
      alignmentScoreGap: span.alignment_score_gap,
      alignmentSource: span.alignment_source,
      anchorConfidence: span.anchor_confidence.tag,
      alignment: span.alignment.map((op) => ({
        kind: op.kind.tag,
        transcriptIndex: op.transcript_index,
        zipaIndex: op.zipa_index,
        transcriptToken: op.transcript_token,
        zipaToken: op.zipa_token,
        cost: op.cost,
      })),
      candidates: span.candidates.map((candidate) => ({
        term: candidate.term,
        aliasText: candidate.alias_text,
        aliasSource: candidate.alias_source.tag,
        candidateNormalized: candidate.candidate_normalized,
        featureSimilarity: candidate.feature_similarity,
        similarityDelta: candidate.similarity_delta,
      })),
    })),
  };
}

function toInspectorData(
  transcript: string,
  words: AlignedWord[],
  phoneticTrace?: PhoneticRescueTrace | null,
): EvalInspectorData {
  const qwenTokens = alignedWordsToTimedTokens(words);
  return {
    transcript,
    transcriptLabel: "BeeML",
    transcriptSource: "transcript",
    qwenAlignment: qwenTokens,
    alignments: {
      timingSource: "qwen-forced-aligner",
      transcript: qwenTokens,
    },
    phoneticTrace,
    prototype: {
      corrected: transcript,
      accepted: [],
      proposals: [],
      sentenceCandidates: [],
    },
  };
}

const micConstraints: MediaTrackConstraints = {
  channelCount: 1,
  sampleRate: 48000,
  sampleSize: 16,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
};

function resampleMonoLinear(
  input: Float32Array,
  inputRate: number,
  outputRate: number,
): Float32Array {
  if (inputRate === outputRate) return input;
  const ratio = inputRate / outputRate;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] ?? input[input.length - 1] ?? 0;
    const b = input[idx + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

export function TranscribeDemoPanel({
  wsUrl,
}: {
  wsUrl: string;
}) {
  const recorder = useAudioRecorder();
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inspectorData, setInspectorData] = useState<EvalInspectorData | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | undefined>(undefined);

  // Streaming state
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [streamCommittedLen, setStreamCommittedLen] = useState(0);
  const streamCleanupRef = useRef<(() => void) | null>(null);

  const handleRecord = useCallback(async () => {
    if (recorder.state === "recording") {
      setStatus("Stopping recording...");
      const blob = await recorder.stop();

      if (audioUrl) URL.revokeObjectURL(audioUrl);
      const nextAudioUrl = URL.createObjectURL(blob);
      setAudioUrl(nextAudioUrl);

      try {
        setStatus("Connecting to BeeML...");
        setError(null);
        const client = await connectBeeMl(wsUrl);

        setStatus("Transcribing...");
        const bytes = new Uint8Array(await blob.arrayBuffer());
        const result = await client.transcribeWav(bytes);
        if (!result.ok) throw new Error(result.error);

        setInspectorData(
          toInspectorData(
            result.value.transcript,
            result.value.words,
            result.value.phonetic_trace
              ? toPhoneticTrace(result.value.phonetic_trace)
              : null,
          ),
        );
        setStatus(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus(null);
      }
    } else {
      setError(null);
      setInspectorData(null);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(undefined);
      await recorder.start();
    }
  }, [audioUrl, recorder, wsUrl]);

  const handleStream = useCallback(async () => {
    if (streaming) {
      // Stop streaming
      streamCleanupRef.current?.();
      streamCleanupRef.current = null;
      setStreaming(false);
      setStatus(null);
      return;
    }

    setError(null);
    setInspectorData(null);
    setStreamText("");
    setStreamCommittedLen(0);

    try {
      setStatus("Connecting...");
      const client = await connectBeeMl(wsUrl);

      // Create channel pairs
      const [audioTx, audioRx] = channel<number[]>();
      const [updatesTx, updatesRx] = channel<Update>();

      // Start the RPC (it runs until we close audioTx)
      const rpcPromise = client.streamTranscribe(audioRx, updatesTx);

      // Receive updates in background
      const receiveLoop = (async () => {
        while (true) {
          const val = await updatesRx.recv();
          if (val === null) break;
          setStreamText(val.text);
          setStreamCommittedLen(val.text.length);
        }
      })();

      // Capture mic and stream chunks
      const stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints });
      const ctx = new AudioContext();
      await ctx.resume();
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, source.channelCount || 1, 1);
      const sink = ctx.createGain();
      sink.gain.value = 0;

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer;
        const len = input.length;
        const channels = input.numberOfChannels || 1;
        const mono = new Float32Array(len);
        for (let ch = 0; ch < channels; ch++) {
          const data = input.getChannelData(ch);
          for (let i = 0; i < len; i++) mono[i] += data[i];
        }
        const scale = 1 / channels;
        for (let i = 0; i < len; i++) mono[i] *= scale;

        // Resample to 16kHz and send
        const resampled = resampleMonoLinear(mono, ctx.sampleRate, 16000);
        audioTx.send(Array.from(resampled)).catch(() => {});
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(ctx.destination);

      setStreaming(true);
      setStatus("Streaming...");

      // Cleanup function
      streamCleanupRef.current = () => {
        processor.disconnect();
        source.disconnect();
        sink.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        ctx.close();
        audioTx.close();

        // Wait for final results
        Promise.all([rpcPromise, receiveLoop]).then(([result]) => {
          if (result && !result.ok) {
            setError(result.error);
          }
        });
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
      setStreaming(false);
    }
  }, [streaming, wsUrl]);

  return (
    <div className="demo-panel">
      <div className="demo-toolbar">
        <button
          className={recorder.state === "recording" ? "danger" : "primary"}
          onClick={handleRecord}
          disabled={streaming || recorder.state === "processing"}
        >
          {recorder.state === "recording" ? "STOP" : "RECORD"}
        </button>

        <button
          className={streaming ? "danger" : "primary"}
          onClick={handleStream}
          disabled={true}
          title="Temporarily unavailable while beeml migrates to the correction RPC surface"
        >
          {streaming ? "STOP STREAM" : "STREAM"}
        </button>

        {status && <span className="status">{status}</span>}
        {error && <span className="error">{error}</span>}
      </div>

      {streaming || streamText ? (
        <div className="demo-stream-output">
          <span className="committed">{streamText.slice(0, streamCommittedLen)}</span>
          <span className="pending">{streamText.slice(streamCommittedLen)}</span>
          {streaming && <span className="cursor" />}
        </div>
      ) : inspectorData ? (
        <EvalInspector data={inspectorData} audioUrl={audioUrl} />
      ) : (
        <div className="demo-empty">
          {recorder.state === "recording"
            ? "Recording... click STOP to transcribe"
            : "Press RECORD for batch transcription, or STREAM for real-time"}
        </div>
      )}
    </div>
  );
}
