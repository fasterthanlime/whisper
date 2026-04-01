import { useState, useRef, useCallback } from "react";

type RecorderState = "idle" | "recording" | "processing";

const micConstraints: MediaTrackConstraints = {
  channelCount: 1,
  sampleRate: 48000,
  sampleSize: 16,
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
};

/** Encode mono Float32 samples into a 16-bit PCM WAV ArrayBuffer. */
function monoSamplesToWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const dataLen = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);
  const ws = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  ws(0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  ws(36, "data");
  v.setUint32(40, dataLen, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function resampleMonoLinear(
  input: Float32Array,
  inputRate: number,
  outputRate: number,
): Float32Array {
  if (inputRate === outputRate) {
    return input;
  }
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

export function useAudioRecorder() {
  const [state, setState] = useState<RecorderState>("idle");
  const ctxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const sinkRef = useRef<GainNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: micConstraints });
    const [track] = stream.getAudioTracks();
    if (track?.applyConstraints) {
      try { await track.applyConstraints(micConstraints); } catch { /* ok */ }
    }

    const ctx = new AudioContext();
    await ctx.resume();
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, source.channelCount || 1, 1);
    const sink = ctx.createGain();
    sink.gain.value = 0;

    chunksRef.current = [];

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
      chunksRef.current.push(mono);
    };

    source.connect(processor);
    processor.connect(sink);
    sink.connect(ctx.destination);

    ctxRef.current = ctx;
    processorRef.current = processor;
    sourceRef.current = source;
    sinkRef.current = sink;
    streamRef.current = stream;
    setState("recording");
  }, []);

  const stop = useCallback(async (): Promise<Blob> => {
    setState("processing");

    // Tear down audio graph
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    sinkRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());

    const sampleRate = ctxRef.current?.sampleRate ?? 48000;
    if (ctxRef.current) {
      await ctxRef.current.close();
    }

    // Merge chunks into a single Float32Array
    const chunks = chunksRef.current;
    const total = chunks.reduce((sum, c) => sum + c.length, 0);
    const mono = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      mono.set(chunk, offset);
      offset += chunk.length;
    }

    const mono16k = resampleMonoLinear(mono, sampleRate, 16000);
    const wavBuf = monoSamplesToWav(mono16k, 16000);
    const blob = new Blob([wavBuf], { type: "audio/wav" });

    ctxRef.current = null;
    processorRef.current = null;
    sourceRef.current = null;
    sinkRef.current = null;
    streamRef.current = null;
    setState("idle");
    return blob;
  }, []);

  return { state, start, stop };
}
