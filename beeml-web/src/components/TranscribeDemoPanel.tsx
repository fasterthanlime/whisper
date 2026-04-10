import { useCallback, useRef, useState } from "react";
import { channel } from "@bearcove/vox-core";
import { connectBeeMl } from "../beeml.generated";
import type {
  AlignedWord,
  SessionSnapshot,
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
    snapshotRevision: trace.snapshot_revision,
    alignedTranscript: trace.aligned_transcript,
    pendingText: trace.pending_text,
    fullTranscript: trace.full_transcript,
    sessionAudioF32: trace.session_audio_f32,
    sessionAudioSampleRateHz: trace.session_audio_sample_rate_hz,
    tailAmbiguity: {
      pendingTokenCount: trace.tail_ambiguity.pending_token_count,
      lowConcentrationCount: trace.tail_ambiguity.low_concentration_count,
      lowMarginCount: trace.tail_ambiguity.low_margin_count,
      volatileTokenCount: trace.tail_ambiguity.volatile_token_count,
      meanConcentration: trace.tail_ambiguity.mean_concentration,
      meanMargin: trace.tail_ambiguity.mean_margin,
      minConcentration: trace.tail_ambiguity.min_concentration,
      minMargin: trace.tail_ambiguity.min_margin,
    },
    worstRawSpanIndex: trace.worst_raw_span_index,
    worstContentfulSpanIndex: trace.worst_contentful_span_index,
    bestRescueSpanIndex: trace.best_rescue_span_index,
    utteranceZipaRaw: trace.utterance_zipa_raw,
    utteranceZipaPhoneSpans: trace.utterance_zipa_phone_spans.map((span) => ({
      tokenId: span.token_id,
      token: span.token,
      startFrame: span.start_frame,
      endFrame: span.end_frame,
      startSec: span.start_sec,
      endSec: span.end_sec,
    })),
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
    asrAlternatives: trace.asr_alternatives.map((token) => ({
      tokenIndex: token.token_index,
      chosenText: token.chosen_text,
      concentration: token.concentration,
      margin: token.margin,
      revision: BigInt(token.revision.toString()),
      alternatives: token.alternatives.map((alternative) => ({
        tokenId: alternative.token_id,
        text: alternative.text,
        logit: alternative.logit,
      })),
    })),
    wordAlignments: trace.word_alignments.map((word) => ({
      wordText: word.word_text,
      tokenStart: word.token_start,
      tokenEnd: word.token_end,
      startSec: word.start_sec,
      endSec: word.end_sec,
      zipaRawPhoneStart: word.zipa_raw_phone_start,
      zipaRawPhoneEnd: word.zipa_raw_phone_end,
      zipaStartSec: word.zipa_start_sec,
      zipaEndSec: word.zipa_end_sec,
      transcriptRaw: word.transcript_raw,
      transcriptNormalized: word.transcript_normalized,
      zipaNormStart: word.zipa_norm_start,
      zipaNormEnd: word.zipa_norm_end,
      zipaRaw: word.zipa_raw,
      zipaNormalized: word.zipa_normalized,
      alignment: word.alignment.map((op) => ({
        kind: op.kind.tag,
        transcriptIndex: op.transcript_index,
        zipaIndex: op.zipa_index,
        transcriptToken: op.transcript_token,
        zipaToken: op.zipa_token,
        cost: op.cost,
      })),
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
      spanClass: span.span_class.tag,
      spanUsefulness: span.span_usefulness.tag,
      zipaRescueEligible: span.zipa_rescue_eligible,
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

function monoSamplesToWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const view = new DataView(buf);
  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataLen, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
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

  // Streaming state
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [streamCommittedLen, setStreamCommittedLen] = useState(0);
  const streamCleanupRef = useRef<(() => void) | null>(null);
  const lastRecordedWavRef = useRef<Uint8Array | null>(null);

  const handleRecord = useCallback(async () => {
    if (recorder.state === "recording") {
      setStatus("Stopping recording...");
      const blob = await recorder.stop();

      try {
        setStatus("Connecting to BeeML...");
        setError(null);
        const client = await connectBeeMl(wsUrl);

        setStatus("Transcribing...");
        const bytes = new Uint8Array(await blob.arrayBuffer());
        lastRecordedWavRef.current = bytes;
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
      await recorder.start();
    }
  }, [recorder, wsUrl]);

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
      const [updatesTx, updatesRx] = channel<SessionSnapshot>();

      // Start the RPC (it runs until we close audioTx)
      const rpcPromise = client.streamTranscribe(audioRx, updatesTx);

      // Receive updates in background
      const receiveLoop = (async () => {
        while (true) {
          const val = await updatesRx.recv();
          if (val === null) break;
          setStreamText(val.full_text);
          setStreamCommittedLen(val.committed_text.length);
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

  const handleSimulateCut = useCallback(
    async (targetCommittedTokens: number) => {
      if (!inspectorData?.phoneticTrace) {
        throw new Error("No phonetic trace available for cut simulation.");
      }

      let wavBytes = lastRecordedWavRef.current;
      if (!wavBytes) {
        const sampleRate =
          inspectorData.phoneticTrace.sessionAudioSampleRateHz || 16000;
        const samples = new Float32Array(inspectorData.phoneticTrace.sessionAudioF32);
        wavBytes = new Uint8Array(monoSamplesToWav(samples, sampleRate));
      }

      setStatus(`Simulating cut @ token ${targetCommittedTokens}...`);
      setError(null);

      const client = await connectBeeMl(wsUrl);
      const result = await client.transcribeWavWithOptions({
        wav_bytes: wavBytes,
        options: {
          chunk_duration: 0.4,
          vad_threshold: 0.5,
          rollback_tokens: 5n,
          commit_token_count: 12n,
          max_tokens_streaming: 32n,
          max_tokens_final: 512n,
          language: [""] as unknown as { 0: string },
          app_id: null,
          rotation_cut_strategy: {
            tag: "ManualTargetCommittedTextTokens",
            value: targetCommittedTokens,
          },
        },
      });
      if (!result.ok) {
        throw new Error(result.error);
      }

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
    },
    [inspectorData, wsUrl],
  );

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
        <EvalInspector
          data={inspectorData}
          wsUrl={wsUrl}
          onSimulateCut={handleSimulateCut}
        />
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
