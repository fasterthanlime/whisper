#[derive(Clone)]
struct WordPlacement {
    text: String,
    chunk_index: usize,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
    quality_label: &'static str,
}

struct SlidingWordPlacement {
    text: String,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
    quality_label: &'static str,
    carried: bool,
    kept: bool,
    bridge: bool,
    cut_word: bool,
}

struct CommittedWordPlacement {
    text: String,
    start_secs: Option<f64>,
    end_secs: Option<f64>,
    quality_label: &'static str,
    second_bin: usize,
}

fn committed_words_equivalent(
    left: &CommittedWordPlacement,
    right: &CommittedWordPlacement,
) -> bool {
    if left.text != right.text {
        return false;
    }
    match (
        left.start_secs,
        left.end_secs,
        right.start_secs,
        right.end_secs,
    ) {
        (Some(left_start), Some(left_end), Some(right_start), Some(right_end)) => {
            (left_start - right_start).abs() <= 0.12 && (left_end - right_end).abs() <= 0.12
        }
        _ => true,
    }
}

fn build_word_placements(
    align_ctx: &mut AlignmentContext,
    chunks: &[ChunkRun],
    samples: &[f32],
) -> Result<Vec<WordPlacement>> {
    let combined_transcript = combine_transcripts(chunks);
    let alignment = build_transcript_alignment(align_ctx, &combined_transcript, samples)?;
    let word_timings = alignment.word_timings();
    let mut next_word = 0usize;
    let mut placements = Vec::new();

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        let chunk_words = sentence_word_tokens(&chunk.transcript);
        for _ in chunk_words {
            let word_timing = word_timings
                .get(next_word)
                .ok_or_else(|| anyhow::anyhow!("missing word timing at index {next_word}"))?;
            let (start_secs, end_secs, quality_label) = match &word_timing.quality {
                bee_transcribe::zipa_align::AlignmentQuality::Aligned {
                    start_secs,
                    end_secs,
                } => (Some(*start_secs), Some(*end_secs), "aligned"),
                bee_transcribe::zipa_align::AlignmentQuality::NoWindow => (None, None, "no-window"),
                bee_transcribe::zipa_align::AlignmentQuality::NoTiming => (None, None, "no-timing"),
            };
            placements.push(WordPlacement {
                text: word_timing.word.to_string(),
                chunk_index,
                start_secs,
                end_secs,
                quality_label,
            });
            next_word += 1;
        }
    }

    Ok(placements)
}

fn write_sliding_window_timed_rollback_html(
    mode_label: &str,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
    wav_path: &Path,
) -> Result<PathBuf> {
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/bee-kv");
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join(format!("{mode_label}.html"));
    let audio_src = file_url_for_path(wav_path)?;
    let mut align_ctx = AlignmentContext::new()?;
    let html = render_sliding_window_timed_rollback_html(
        &mut align_ctx,
        mode_label,
        window_runs,
        samples,
        duration_secs,
        &audio_src,
    )?;
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

fn write_committed_timeline_html(
    mode_label: &str,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
    wav_path: &Path,
) -> Result<PathBuf> {
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/bee-kv");
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join(format!("{mode_label}.html"));
    let audio_src = file_url_for_path(wav_path)?;
    let mut align_ctx = AlignmentContext::new()?;
    let words = collect_committed_word_placements(&mut align_ctx, window_runs, samples)?;
    let html = render_committed_timeline_html(mode_label, duration_secs, &words, &audio_src);
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

fn render_sliding_window_timed_rollback_html(
    align_ctx: &mut AlignmentContext,
    mode_label: &str,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
    total_duration_secs: f64,
    audio_src: &str,
) -> Result<String> {
    let width_px = 1100.0;
    let row_height_px = 110.0;
    let mut rows = String::new();

    for (row_index, run) in window_runs.iter().enumerate() {
        let chunk = &run.chunk_run;
        let chunk_samples = &samples[chunk.start_sample..chunk.end_sample];
        let words = build_window_word_placements(align_ctx, run, chunk_samples)?;
        rows.push_str(&render_sliding_window_row(
            width_px,
            row_height_px,
            row_index,
            chunk,
            run.rollback.as_ref(),
            run.replayed_prefix.as_ref(),
            &words,
            total_duration_secs,
            audio_src,
        ));
    }

    Ok(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>bee-kv {mode_label}</title><style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#111315;color:#ece7dc;padding:24px;color-scheme:dark;}}\
.legend{{margin-bottom:18px;font-size:13px;color:#b9b3a7;}}\
.audio-panel{{margin:0 0 20px 0;padding:12px 14px;border:1px solid #4b4f56;background:#181c20;width:{width_px}px;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.audio-title{{font-weight:700;margin:0 0 8px 0;}}\
.audio-player{{width:100%;margin:6px 0 0 0;color-scheme:dark;accent-color:#d97706;}}\
.row{{margin-bottom:28px;}}\
.row-title{{font-weight:700;margin:0 0 6px 0;}}\
.row-meta{{margin:0 0 8px 0;font-size:13px;color:#b0aa9d;}}\
.row-audio{{margin:0 0 8px 0;display:flex;align-items:center;gap:10px;}}\
.row-audio audio{{width:420px;max-width:100%;color-scheme:dark;accent-color:#d97706;}}\
.transcript-line{{width:{width_px}px;margin:0 0 8px 0;font-size:13px;line-height:1.5;color:#ded7ca;}}\
.timeline{{width:{width_px}px;border:1px solid #4b4f56;background:#181c20;position:relative;padding:12px 0;margin-bottom:8px;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.track{{position:relative;width:{width_px}px;height:{row_height_px}px;border-top:1px solid #343940;border-bottom:1px solid #343940;background:linear-gradient(180deg,#1b2024,#14181c);overflow:hidden;}}\
.word{{text-align:center;vertical-align:middle;position:absolute;height:2em;padding:4px 2px;border-radius:4px;border:1px solid #7a6d56;white-space:nowrap;overflow:visible;font-size:12px;box-sizing:border-box;cursor:default;font-family:\"SF Pro\", serif;box-shadow:0 3px 10px rgba(0,0,0,0.28);}}\
.word.carried{{background:#544775;border-color:#aa9cf0;color:#f0ebff;}}\
.word.kept{{background:#183b2a;border-color:#62d596;color:#dffff0;font-weight:700;box-shadow:0 0 0 2px rgba(98,213,150,0.24),0 3px 10px rgba(0,0,0,0.28);}}\
.word.bridge{{background:#4b3c12;border-color:#e4bd4f;color:#fff1bf;}}\
.word.rolled{{background:#4d2528;border-color:#db7f88;color:#ffe0e3;}}\
.word.cut-word{{outline:3px solid rgba(117,173,255,0.6);outline-offset:1px;border-style:dashed;}}\
.word.no-window,.word.no-timing{{background:#3b352c;border-color:#908672;color:#f1e8d5;height:22px;font-size:11px;}}\
.search-range{{position:absolute;height:12px;border-bottom:2px solid #7e8793;border-left:2px solid #7e8793;border-right:2px solid #7e8793;border-bottom-left-radius:8px;border-bottom-right-radius:8px;background:rgba(126,135,147,0.10);pointer-events:none;}}\
.search-range-label{{position:absolute;transform:translateX(-50%);font-size:11px;color:#9ba4af;font-weight:700;pointer-events:none;}}\
.cut{{position:absolute;top:0;width:2px;height:{row_height_px}px;background:#5ba2ff;}}\
.cut-label{{position:absolute;transform:translateX(6px);font-size:11px;color:#7fb6ff;font-weight:700;white-space:nowrap;}}\
.cut-label.cut-label-cut{{top:2px;}}\
.cut-label.cut-label-target{{top:18px;}}\
.cut-label.cut-label-bridge{{top:2px;}}\
.window-start,.window-end{{position:absolute;top:0;width:1px;height:{row_height_px}px;background:#5b616b;}}\
.window-label{{position:absolute;bottom:2px;transform:translateX(4px);font-size:11px;color:#9299a4;}}\
.playhead{{position:absolute;top:0;width:2px;height:{row_height_px}px;background:#d97706;pointer-events:none;display:none;}}\
.axis{{display:flex;justify-content:space-between;font-size:12px;color:#9299a4;margin-top:6px;}}\
.word:hover::after{{content:attr(data-full-word);position:absolute;left:0;top:-28px;background:#f6f1e5;color:#111315;padding:2px 6px;border-radius:4px;white-space:nowrap;z-index:10;font-size:11px;line-height:16px;box-shadow:0 2px 6px rgba(0,0,0,0.35);}}\
</style></head><body><h1>bee-kv {mode_label}</h1><div class=\"audio-panel\"><div class=\"audio-title\">Full Recording</div><div>Source: {audio_src}</div><audio id=\"master-audio\" class=\"audio-player\" controls preload=\"metadata\" src=\"{audio_src}\"></audio></div><p class=\"legend\">Each row is one decode window. Purple words are prompt text replayed into that row. Green words are KV-kept generated words. Yellow words are bridge-region generated words. Red words are re-decoded tail words. Blue line marks the keep cut. Brown line marks the bridge cut. Orange line is the current playhead.</p>{rows}<script>\
const masterAudio = document.getElementById('master-audio');\
const chunkAudios = Array.from(document.querySelectorAll('audio[data-window-start]'));\
const allAudios = [masterAudio, ...chunkAudios].filter(Boolean);\
let rafId = null;\
let activeAudio = null;\
function pauseOthers(active){{ for (const audio of allAudios) {{ if (audio !== active) audio.pause(); }} }}\
function updatePlayheads(time){{\
  document.querySelectorAll('[data-row-start]').forEach((row) => {{\
    const start = Number(row.dataset.rowStart);\
    const end = Number(row.dataset.rowEnd);\
    const duration = end - start;\
    const playhead = row.querySelector('.playhead');\
    if (!playhead) return;\
    if (time < start || time > end || duration <= 0) {{ playhead.style.display = 'none'; return; }}\
    const frac = (time - start) / duration;\
    playhead.style.display = 'block';\
    playhead.style.left = `${{Math.max(0, Math.min(1, frac)) * 100}}%`;\
  }});\
}}\
function hidePlayheads(){{\
  document.querySelectorAll('[data-row-start] .playhead').forEach((playhead) => {{\
    playhead.style.display = 'none';\
  }});\
}}\
function syncPlayheads(audio){{ updatePlayheads(audio?.currentTime || 0); }}\
function stopTracking(){{\
  if (rafId !== null) cancelAnimationFrame(rafId);\
  rafId = null;\
  activeAudio = null;\
  hidePlayheads();\
}}\
function tick(){{\
  if (!activeAudio || activeAudio.paused || activeAudio.ended) {{ stopTracking(); return; }}\
  if (chunkAudios.includes(activeAudio)) {{\
    const end = Number(activeAudio.dataset.windowEnd);\
    if ((activeAudio.currentTime || 0) >= end) {{\
      activeAudio.currentTime = end;\
      activeAudio.pause();\
      stopTracking();\
      return;\
    }}\
  }}\
  syncPlayheads(activeAudio);\
  rafId = requestAnimationFrame(tick);\
}}\
function startTracking(audio){{\
  pauseOthers(audio);\
  activeAudio = audio;\
  if (rafId !== null) cancelAnimationFrame(rafId);\
  syncPlayheads(audio);\
  rafId = requestAnimationFrame(tick);\
}}\
masterAudio.addEventListener('play', () => startTracking(masterAudio));\
masterAudio.addEventListener('pause', () => {{ if (activeAudio === masterAudio) stopTracking(); }});\
masterAudio.addEventListener('ended', () => {{ if (activeAudio === masterAudio) stopTracking(); }});\
masterAudio.addEventListener('seeking', () => syncPlayheads(masterAudio));\
chunkAudios.forEach((audio) => {{\
  const start = Number(audio.dataset.windowStart);\
  const end = Number(audio.dataset.windowEnd);\
  audio.addEventListener('play', () => {{\
    if (audio.currentTime < start || audio.currentTime >= end) audio.currentTime = start;\
    startTracking(audio);\
  }});\
  audio.addEventListener('seeking', () => {{\
    if (audio.currentTime < start) audio.currentTime = start;\
    if (audio.currentTime > end) audio.currentTime = end;\
    if (activeAudio === audio) syncPlayheads(audio);\
  }});\
  audio.addEventListener('pause', () => {{ if (activeAudio === audio) stopTracking(); }});\
  audio.addEventListener('ended', () => {{ if (activeAudio === audio) stopTracking(); }});\
}});\
updatePlayheads(0);\
</script></body></html>"
    ))
}

fn collect_committed_word_placements(
    align_ctx: &mut AlignmentContext,
    window_runs: &[SlidingWindowRun],
    samples: &[f32],
) -> Result<Vec<CommittedWordPlacement>> {
    let mut placements: Vec<CommittedWordPlacement> = Vec::new();

    for run in window_runs {
        let chunk = &run.chunk_run;
        let chunk_samples = &samples[chunk.start_sample..chunk.end_sample];
        let words = build_window_word_placements(align_ctx, run, chunk_samples)?;
        let window_start_secs = chunk.start_sample as f64 / SAMPLE_RATE as f64;
        let keep_word_count = if let Some(rollback) = &run.rollback {
            sentence_word_tokens(&rollback.kept_text).len()
        } else {
            words.len()
        };
        if keep_word_count == 0 {
            continue;
        }
        let mut candidate = Vec::new();
        for word in words.into_iter().take(keep_word_count) {
            let start_secs = word.start_secs.map(|start| window_start_secs + start);
            let end_secs = word.end_secs.map(|end| window_start_secs + end);
            let second_bin = start_secs
                .map(|start| start.floor() as usize)
                .unwrap_or_else(|| window_start_secs.floor() as usize);
            candidate.push(CommittedWordPlacement {
                text: word.text,
                start_secs,
                end_secs,
                quality_label: word.quality_label,
                second_bin,
            });
        }
        let max_overlap = placements.len().min(candidate.len());
        let overlap = (0..=max_overlap)
            .rev()
            .find(|&count| {
                placements[placements.len().saturating_sub(count)..]
                    .iter()
                    .zip(candidate.iter().take(count))
                    .all(|(left, right)| committed_words_equivalent(left, right))
            })
            .unwrap_or(0);
        placements.extend(candidate.into_iter().skip(overlap));
    }

    Ok(placements)
}

fn render_committed_timeline_html(
    mode_label: &str,
    duration_secs: f64,
    words: &[CommittedWordPlacement],
    audio_src: &str,
) -> String {
    let px_per_sec = 100.0_f64;
    let width_px = (duration_secs * px_per_sec).ceil();
    let row_height_px = 132.0;
    let transcript_line = words
        .iter()
        .map(|word| html_escape(&word.text))
        .collect::<Vec<_>>()
        .join(" ");
    let total_seconds = duration_secs.ceil() as usize;
    let mut second_bands = String::new();
    let mut second_markers = String::new();
    for second in 0..=total_seconds {
        let x = ((second as f64) / duration_secs.min(duration_secs.max(1.0))) * width_px;
        if second < total_seconds {
            let left = (second as f64 / duration_secs) * width_px;
            let right = (((second + 1) as f64).min(duration_secs) / duration_secs) * width_px;
            let width = (right - left).max(0.0);
            let hue = (second * 47) % 360;
            second_bands.push_str(&format!(
                "<div class=\"second-band\" style=\"left:{left:.1}px;width:{width:.1}px;background:hsl({hue} 55% 90% / 0.55)\"></div>"
            ));
        }
        if second as f64 <= duration_secs {
            second_markers.push_str(&format!(
                "<div class=\"second-marker\" style=\"left:{x:.1}px\"></div><div class=\"second-label\" style=\"left:{x:.1}px\">{second}s</div>"
            ));
        }
    }

    let mut word_divs = String::new();
    let mut fallback_x = 0.0;
    let mut lane_end_x = [0.0_f64; 3];
    let lane_tops = [20.0_f64, 54.0_f64, 88.0_f64];
    for word in words {
        let hue = (word.second_bin * 47) % 360;
        let class = format!("word {}", word.quality_label);
        let (left, width, lane_index) = match (word.start_secs, word.end_secs) {
            (Some(start), Some(end)) => {
                let left = (start / duration_secs) * width_px;
                let width = ((end - start).max(0.08) / duration_secs) * width_px;
                let mut lane_index = 0usize;
                while lane_index + 1 < lane_end_x.len() && lane_end_x[lane_index] > left {
                    lane_index += 1;
                }
                if lane_end_x[lane_index] > left && lane_index == lane_end_x.len() - 1 {
                    lane_index = lane_end_x
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                }
                lane_end_x[lane_index] = left + width + 6.0;
                (left, width, lane_index)
            }
            _ => {
                let left = lane_end_x.iter().copied().fold(fallback_x, f64::max);
                fallback_x = left + 90.0;
                (left, 84.0, 2)
            }
        };
        let top = lane_tops[lane_index.min(lane_tops.len() - 1)];
        let text = html_escape(&word.text);
        word_divs.push_str(&format!(
            "<div class=\"{class}\" style=\"left:{left:.1}px;top:{top:.1}px;width:{width:.1}px;background:hsl({hue} 58% 82%);border-color:hsl({hue} 38% 42%);\" title=\"committed @ {start:.2}-{end:.2}s\" data-full-word=\"{text}\">{text}</div>",
            start = word.start_secs.unwrap_or(0.0),
            end = word.end_secs.unwrap_or(0.0),
        ));
    }

    let word_timings_js = words
        .iter()
        .filter_map(|w| {
            let s = w.start_secs?;
            let e = w.end_secs?;
            let t = w.text.replace('\\', "\\\\").replace('"', "\\\"");
            Some(format!("{{t:\"{t}\",s:{s:.3},e:{e:.3}}}"))
        })
        .collect::<Vec<_>>()
        .join(",");

    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>bee-kv {mode_label}</title><style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#111315;color:#ece7dc;padding:24px;color-scheme:dark;}}\
.legend{{margin-bottom:18px;font-size:13px;color:#b9b3a7;}}\
.audio-panel{{margin:0 0 20px 0;padding:12px 14px;border:1px solid #4b4f56;background:#181c20;max-width:900px;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.audio-title{{font-weight:700;margin:0 0 8px 0;}}\
.audio-player{{width:100%;margin:6px 0 0 0;color-scheme:dark;accent-color:#d97706;}}\
#current-word-display{{font-family:\"SF Pro Display\",\"SF Pro\",ui-sans-serif,-apple-system,sans-serif;font-size:80px;font-weight:700;height:110px;display:flex;align-items:center;justify-content:center;color:#f6f1e5;margin:20px 0;letter-spacing:-0.01em;flex-shrink:0;}}\
.timeline-scroll{{overflow-x:auto;margin-bottom:8px;}}\
.transcript-line{{width:{width_px}px;margin:0 0 8px 0;font-size:13px;line-height:1.5;color:#ded7ca;}}\
.timeline{{width:{width_px}px;border:1px solid #4b4f56;background:#181c20;position:relative;padding:12px 0;box-shadow:0 10px 30px rgba(0,0,0,0.22);}}\
.track{{position:relative;width:{width_px}px;height:{row_height_px}px;border-top:1px solid #343940;border-bottom:1px solid #343940;background:linear-gradient(180deg,#1b2024,#14181c);overflow:hidden;}}\
.second-band{{position:absolute;top:0;height:{row_height_px}px;pointer-events:none;mix-blend-mode:screen;}}\
.second-marker{{position:absolute;top:0;width:1px;height:{row_height_px}px;background:#5b616b;}}\
.second-label{{position:absolute;top:2px;transform:translateX(4px);font-size:11px;color:#9299a4;}}\
.playhead{{position:absolute;top:0;width:2px;height:{row_height_px}px;background:#d97706;pointer-events:none;display:none;}}\
.word{{text-align:center;vertical-align:middle;position:absolute;height:2em;padding:4px 2px;border-radius:4px;border:1px solid #7a6d56;white-space:nowrap;overflow:visible;font-size:12px;box-sizing:border-box;cursor:default;font-family:\"SF Pro\", serif;color:#101214;box-shadow:0 3px 10px rgba(0,0,0,0.28);}}\
.word.no-window,.word.no-timing{{background:#3b352c !important;border-color:#908672 !important;color:#f1e8d5 !important;height:22px;font-size:11px;}}\
.axis{{display:flex;justify-content:space-between;font-size:12px;color:#9299a4;margin-top:6px;}}\
.word:hover::after{{content:attr(data-full-word);position:absolute;left:0;top:-28px;background:#f6f1e5;color:#111315;padding:2px 6px;border-radius:4px;white-space:nowrap;z-index:10;font-size:11px;line-height:16px;box-shadow:0 2px 6px rgba(0,0,0,0.35);}}\
</style></head><body>\
<h1>bee-kv {mode_label}</h1>\
<div class=\"audio-panel\"><div class=\"audio-title\">Full Recording</div><div>Source: {audio_src}</div><audio id=\"master-audio\" class=\"audio-player\" controls preload=\"metadata\" src=\"{audio_src}\"></audio></div>\
<p class=\"legend\">Committed words only. Each word keeps the exact timing it had when it was marked green in its source row. Background bands and box colors change every one-second interval; no extra re-alignment is performed.</p>\
<div id=\"current-word-display\"></div>\
<div class=\"timeline-scroll\">\
<div class=\"transcript-line\">{transcript_line}</div>\
<div class=\"timeline\" data-row-start=\"0\" data-row-end=\"{duration_secs:.6}\">\
<div class=\"track\">{second_bands}{second_markers}<div class=\"playhead\"></div>{word_divs}</div>\
<div class=\"axis\"><span>0.00s</span><span>{duration_secs:.2}s total</span></div>\
</div></div>\
<script>\
const wordTimings=[{word_timings_js}];\
const trackWidth={width_px};\
const audio=document.getElementById('master-audio');\
const row=document.querySelector('[data-row-start]');\
const playhead=row?.querySelector('.playhead');\
const currentWordEl=document.getElementById('current-word-display');\
const scrollEl=document.querySelector('.timeline-scroll');\
let rafId=null;\
function findWord(t){{for(let i=0;i<wordTimings.length;i++){{const w=wordTimings[i];if(t>=w.s&&t<=w.e)return w.t;}}return '';}}\
function updatePlayhead(time){{\
  if(!row||!playhead)return;\
  const start=Number(row.dataset.rowStart);\
  const end=Number(row.dataset.rowEnd);\
  const dur=end-start;\
  if(time<start||time>end||dur<=0){{playhead.style.display='none';return;}}\
  const frac=(time-start)/dur;\
  const px=Math.max(0,Math.min(1,frac))*trackWidth;\
  playhead.style.display='block';\
  playhead.style.left=`${{px}}px`;\
  if(scrollEl){{const cw=scrollEl.offsetWidth;scrollEl.scrollLeft=Math.max(0,px-cw/2);}}\
  if(currentWordEl)currentWordEl.textContent=findWord(time);\
}}\
function hidePlayhead(){{if(playhead)playhead.style.display='none';}}\
function syncPlayhead(){{updatePlayhead(audio.currentTime||0);}}\
function stopTracking(){{if(rafId!==null)cancelAnimationFrame(rafId);rafId=null;hidePlayhead();if(currentWordEl)currentWordEl.textContent='';}}\
function tick(){{if(audio.paused||audio.ended){{stopTracking();return;}}syncPlayhead();rafId=requestAnimationFrame(tick);}}\
audio.addEventListener('play',()=>{{if(rafId!==null)cancelAnimationFrame(rafId);syncPlayhead();rafId=requestAnimationFrame(tick);}});\
audio.addEventListener('pause',stopTracking);\
audio.addEventListener('ended',stopTracking);\
audio.addEventListener('seeking',()=>updatePlayhead(audio.currentTime||0));\
updatePlayhead(0);\
</script></body></html>",
    )
}

fn render_sliding_window_row(
    width_px: f64,
    row_height_px: f64,
    row_index: usize,
    chunk: &ChunkRun,
    rollback: Option<&WindowRollbackDecision>,
    replayed_prefix: Option<&CarriedBridge>,
    words: &[SlidingWordPlacement],
    total_duration_secs: f64,
    audio_src: &str,
) -> String {
    let transcript_line = html_escape(&chunk.transcript);
    let mut markers = String::new();
    let window_start_secs = chunk.start_sample as f64 / SAMPLE_RATE as f64;
    let window_end_secs = chunk.end_sample as f64 / SAMPLE_RATE as f64;
    let window_duration_secs = (chunk.end_sample - chunk.start_sample) as f64 / SAMPLE_RATE as f64;

    markers.push_str(&format!(
        "<div class=\"window-start\" style=\"left:0px\"></div><div class=\"window-end\" style=\"left:{:.1}px\"></div><div class=\"window-label\" style=\"left:0px\">{:.2}s</div><div class=\"window-label\" style=\"left:{:.1}px\">{:.2}s</div>",
        width_px,
        window_start_secs,
        width_px,
        window_end_secs
    ));
    markers.push_str("<div class=\"playhead\"></div>");
    if let Some(rollback) = rollback {
        let search_start_x = (rollback.keep_boundary_debug.earliest_candidate_secs
            / window_duration_secs)
            * width_px;
        let search_end_x = (rollback.target_keep_until_secs / window_duration_secs) * width_px;
        let search_width = (search_end_x - search_start_x).max(0.0);
        let search_label_x = search_start_x + (search_width / 2.0);
        markers.push_str(&format!(
            "<div class=\"search-range\" style=\"left:{search_start_x:.1}px;top:{:.1}px;width:{search_width:.1}px\"></div><div class=\"search-range-label\" style=\"left:{search_label_x:.1}px;top:{:.1}px\">search</div>",
            row_height_px - 14.0,
            row_height_px - 28.0,
        ));
        if let Some(keep_until_secs) = rollback.keep_until_secs {
            let cut_x = (keep_until_secs / window_duration_secs) * width_px;
            markers.push_str(&format!(
                "<div class=\"cut\" style=\"left:{cut_x:.1}px\"></div><div class=\"cut-label cut-label-cut\" style=\"left:{cut_x:.1}px\">cut @{:.2}s</div>",
                window_start_secs + keep_until_secs
            ));
        }
        let target_cut_x = (rollback.target_keep_until_secs / window_duration_secs) * width_px;
        markers.push_str(&format!(
            "<div class=\"cut\" style=\"left:{target_cut_x:.1}px;background:#5a5a5a;opacity:0.55\"></div><div class=\"cut-label cut-label-target\" style=\"left:{target_cut_x:.1}px;color:#5a5a5a\">target @{:.2}s</div>",
            window_start_secs + rollback.target_keep_until_secs
        ));
        if let Some(replay_until_secs) = rollback.replay_until_secs {
            let replay_x = (replay_until_secs / window_duration_secs) * width_px;
            markers.push_str(&format!(
                "<div class=\"cut\" style=\"left:{replay_x:.1}px;background:#8b5e1a\"></div><div class=\"cut-label cut-label-bridge\" style=\"left:{replay_x:.1}px;color:#8b5e1a\">bridge @{:.2}s</div>",
                window_start_secs + replay_until_secs
            ));
        }
    }

    let mut word_divs = String::new();
    let mut fallback_x = 0.0;
    let mut lane_end_x = [0.0_f64; 3];
    let lane_tops = [16.0_f64, 46.0_f64, 76.0_f64];
    for word in words {
        let segment_class = if word.carried {
            "carried"
        } else if word.kept {
            "kept"
        } else if word.bridge {
            "bridge"
        } else {
            "rolled"
        };
        let cut_word_class = if word.cut_word { " cut-word" } else { "" };
        let class = format!(
            "word {segment_class} {}{cut_word_class}",
            word.quality_label
        );
        let (left, width, lane_index) = match (word.start_secs, word.end_secs) {
            (Some(start), Some(end)) => {
                let left = (start / window_duration_secs) * width_px;
                let width = ((end - start).max(0.08) / window_duration_secs) * width_px;
                let mut lane_index = 0usize;
                while lane_index + 1 < lane_end_x.len() && lane_end_x[lane_index] > left {
                    lane_index += 1;
                }
                if lane_end_x[lane_index] > left && lane_index == lane_end_x.len() - 1 {
                    lane_index = lane_end_x
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                }
                lane_end_x[lane_index] = left + width + 6.0;
                (left, width, lane_index)
            }
            _ => {
                let left = lane_end_x.iter().copied().fold(fallback_x, f64::max);
                fallback_x = left + 90.0;
                (left, 84.0, 2)
            }
        };
        let top = lane_tops[lane_index.min(lane_tops.len() - 1)];
        word_divs.push_str(&format!(
            "<div class=\"{class}\" style=\"left:{left:.1}px;top:{top:.1}px;width:{width:.1}px\" title=\"window {start:.2}-{end:.2}s: {text}\" data-full-word=\"{text}\">{text}</div>",
            start = window_start_secs,
            end = window_end_secs,
            text = html_escape(&word.text)
        ));
    }

    let meta = if let Some(rollback) = rollback {
        let chosen_boundary = rollback
            .keep_boundary_debug
            .chosen_word
            .as_ref()
            .map(|word| {
                format!(
                    "{} [{:.2}-{:.2}s]",
                    html_escape(&word.text),
                    window_start_secs + word.start_secs,
                    window_start_secs + word.end_secs
                )
            })
            .unwrap_or_else(|| "none".to_string());
        format!(
            "audio {:.2}s..{:.2}s | replayed_prefix={} | kept_text={} | bridge_text={} | kept_tokens={} | bridge_tokens={} | rollback_position={} | keep_policy={} | keep_target={:.2}s | keep_cut={:.2}s | keep_search=[{:.2}s..{:.2}s] | min_keep={:.2}s | snapped={} | keep_word={}",
            window_start_secs,
            window_end_secs,
            html_escape(replayed_prefix.map(|p| p.text.as_str()).unwrap_or("none")),
            html_escape(&rollback.kept_text),
            html_escape(rollback.bridge_text.as_deref().unwrap_or("none")),
            rollback.kept_token_count,
            rollback.bridge_token_ids.len(),
            rollback.rollback_position,
            rollback.keep_boundary_policy.as_str(),
            window_start_secs + rollback.target_keep_until_secs,
            window_start_secs + rollback.keep_until_secs.unwrap_or(0.0),
            window_start_secs + rollback.keep_boundary_debug.earliest_candidate_secs,
            window_start_secs + rollback.target_keep_until_secs,
            window_start_secs + rollback.keep_boundary_debug.min_keep_secs,
            if rollback.keep_boundary_debug.snapped {
                "yes"
            } else {
                "no"
            },
            chosen_boundary
        )
    } else {
        format!(
            "audio {:.2}s..{:.2}s | final window",
            window_start_secs, window_end_secs
        )
    };

    format!(
        "<section class=\"row\" data-row-start=\"{:.6}\" data-row-end=\"{:.6}\"><div class=\"row-title\">{}</div><div class=\"row-meta\">{}</div><div class=\"row-audio\"><span>Chunk Audio</span><audio id=\"chunk-audio-{}\" controls preload=\"metadata\" src=\"{}\" data-window-start=\"{:.6}\" data-window-end=\"{:.6}\"></audio></div><div class=\"transcript-line\">{}</div><div class=\"timeline\"><div class=\"track\">{}{}</div><div class=\"axis\"><span>0.00s</span><span>{:.2}s window</span><span>{:.2}s total</span></div></div></section>",
        window_start_secs,
        window_end_secs,
        html_escape(&chunk.label),
        meta,
        row_index,
        audio_src,
        window_start_secs,
        window_end_secs,
        transcript_line,
        markers,
        word_divs,
        window_duration_secs,
        total_duration_secs
    )
}

fn build_window_word_placements(
    align_ctx: &mut AlignmentContext,
    run: &SlidingWindowRun,
    chunk_samples: &[f32],
) -> Result<Vec<SlidingWordPlacement>> {
    let generated_transcript = normalized_transcript(&run.chunk_run.transcript);
    let replayed_prefix = run
        .replayed_prefix
        .as_ref()
        .map(|prefix| normalized_transcript(&prefix.text))
        .filter(|text| !text.is_empty());
    if generated_transcript.is_empty() && replayed_prefix.is_none() {
        return Ok(Vec::new());
    }

    let combined_transcript = match (replayed_prefix, generated_transcript.is_empty()) {
        (Some(prefix), false) => format!("{prefix} {generated_transcript}"),
        (Some(prefix), true) => prefix.to_string(),
        (None, false) => generated_transcript.to_string(),
        (None, true) => String::new(),
    };
    let alignment = build_transcript_alignment(align_ctx, &combined_transcript, chunk_samples)?;
    let word_timings = alignment.word_timings();
    let carried_word_count = replayed_prefix
        .map(sentence_word_tokens)
        .map(|words| words.len())
        .unwrap_or(0);
    let kept_word_count = run
        .rollback
        .as_ref()
        .map_or(word_timings.len(), |r| r.kept_word_count);
    let replay_word_count = run.rollback.as_ref().map_or(word_timings.len(), |r| {
        let bridge_words = r
            .bridge_text
            .as_ref()
            .map(|text| sentence_word_tokens(text).len())
            .unwrap_or(0);
        r.kept_word_count + bridge_words
    });
    let chosen_cut = run
        .rollback
        .as_ref()
        .and_then(|r| r.keep_boundary_debug.chosen_word.as_ref());

    Ok(word_timings
        .into_iter()
        .enumerate()
        .map(|(index, word_timing)| {
            let (start_secs, end_secs, quality_label) = match word_timing.quality {
                bee_transcribe::zipa_align::AlignmentQuality::Aligned {
                    start_secs,
                    end_secs,
                } => (Some(start_secs), Some(end_secs), "aligned"),
                bee_transcribe::zipa_align::AlignmentQuality::NoWindow => (None, None, "no-window"),
                bee_transcribe::zipa_align::AlignmentQuality::NoTiming => (None, None, "no-timing"),
            };
            let carried = index < carried_word_count;
            let cut_word = match (&chosen_cut, start_secs, end_secs) {
                (Some(chosen), Some(start), Some(end)) => {
                    word_timing.word == chosen.text
                        && (start - chosen.start_secs).abs() < 0.000_1
                        && (end - chosen.end_secs).abs() < 0.000_1
                }
                _ => false,
            };
            SlidingWordPlacement {
                text: word_timing.word.to_string(),
                start_secs,
                end_secs,
                quality_label,
                carried,
                kept: index < kept_word_count,
                bridge: index >= kept_word_count && index < replay_word_count,
                cut_word,
            }
        })
        .collect())
}

fn write_chunk_segment_merge_rollback_html(
    baseline_runs: &[ChunkRun],
    replay_runs: &[ChunkRun],
    samples: &[f32],
) -> Result<PathBuf> {
    let mut align_ctx = AlignmentContext::new()?;
    let baseline_words = build_word_placements(&mut align_ctx, baseline_runs, samples)?;
    let replay_words = build_word_placements(&mut align_ctx, replay_runs, samples)?;
    let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/bee-kv");
    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let out_path = out_dir.join("chunk-segment-merge-rollback.html");
    let html = render_word_timeline_html(
        duration_secs,
        baseline_runs,
        replay_runs,
        &baseline_words,
        &replay_words,
    );
    fs::write(&out_path, html).with_context(|| format!("writing {}", out_path.display()))?;
    Ok(out_path)
}

fn render_word_timeline_html(
    duration_secs: f64,
    baseline_runs: &[ChunkRun],
    replay_runs: &[ChunkRun],
    baseline_words: &[WordPlacement],
    replay_words: &[WordPlacement],
) -> String {
    let width_px = 1400.0;
    let row_height_px = 132.0;
    let baseline_row = render_word_row(
        "Baseline",
        width_px,
        row_height_px,
        duration_secs,
        baseline_runs,
        baseline_words,
    );
    let replay_row = render_word_row(
        "Replay",
        width_px,
        row_height_px,
        duration_secs,
        replay_runs,
        replay_words,
    );
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>bee-kv rollback word timeline</title><style>\
body{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f7f4ec;color:#1d1b19;padding:24px;}}\
.timeline{{width:{width_px}px;border:1px solid #b9b09f;background:#fffdf8;position:relative;padding:12px 0;margin-bottom:28px;}}\
.row-title{{font-weight:700;margin:0 0 8px 0;}}\
.transcript-line{{width:{width_px}px;margin:0 0 8px 0;font-size:13px;line-height:1.5;color:#3b352d;}}\
.chunk-divider{{color:#8a7f6a;padding:0 6px;}}\
.track{{position:relative;width:{width_px}px;height:{row_height_px}px;border-top:1px solid #d6cfbf;border-bottom:1px solid #d6cfbf;background:linear-gradient(180deg,#fffdf8,#f4efe3);overflow:hidden;}}\
.word{{text-align:center;vertical-align:middle;position:absolute;height:2em;padding:4px 2px;border-radius:4px;border:1px solid #7a6d56;background:#efe2b8;white-space:nowrap;overflow:visible;font-size:12px;box-sizing:border-box;cursor:default;font-family:\"SF Pro\", serif;}}\
.word.chunk-0{{background:#e7d9a8;}} .word.chunk-1{{background:#cfdcc8;}} .word.chunk-2{{background:#d7d0ea;}}\
.word.no-window,.word.no-timing{{background:#e7c2c2;border-color:#9f5d5d;height:22px;font-size:11px;}}\
.boundary{{position:absolute;top:0;width:1px;height:{row_height_px}px;background:#9d9483;}}\
.boundary-label{{position:absolute;top:2px;transform:translateX(4px);font-size:11px;color:#6d6457;}}\
.axis{{display:flex;justify-content:space-between;font-size:12px;color:#6d6457;margin-top:6px;}}\
.legend{{margin-bottom:16px;font-size:13px;color:#514a41;}}\
.word:hover::after{{content:attr(data-full-word);position:absolute;left:0;top:-28px;background:#1d1b19;color:#fffdf8;padding:2px 6px;border-radius:4px;white-space:nowrap;z-index:10;font-size:11px;line-height:16px;box-shadow:0 2px 6px rgba(0,0,0,0.18);}}\
</style></head><body><h1>bee-kv rollback word timeline</h1><p class=\"legend\">Word boxes are positioned by ZIPA-derived word timings. Vertical lines mark chunk ends.</p>{baseline_row}{replay_row}</body></html>"
    )
}

fn render_word_row(
    title: &str,
    width_px: f64,
    _row_height_px: f64,
    duration_secs: f64,
    runs: &[ChunkRun],
    words: &[WordPlacement],
) -> String {
    let transcript_line = runs
        .iter()
        .map(|chunk| html_escape(&chunk.transcript))
        .collect::<Vec<_>>()
        .join("<span class=\"chunk-divider\">|</span>");
    let mut boundaries = String::new();
    for run in runs {
        let x = ((run.end_sample as f64 / SAMPLE_RATE as f64) / duration_secs) * width_px;
        let ms = (run.end_sample * 1000) / SAMPLE_RATE as usize;
        boundaries.push_str(&format!(
            "<div class=\"boundary\" style=\"left:{x:.1}px\"></div><div class=\"boundary-label\" style=\"left:{x:.1}px\">{ms}ms</div>"
        ));
    }

    let mut word_divs = String::new();
    let mut fallback_x = 0.0;
    let mut lane_end_x = [0.0_f64; 3];
    let lane_tops = [20.0_f64, 54.0_f64, 88.0_f64];
    for word in words {
        let class = format!(
            "word chunk-{} {}",
            word.chunk_index.min(2),
            word.quality_label
        );
        let (left, width, lane_index) = match (word.start_secs, word.end_secs) {
            (Some(start), Some(end)) => {
                let left = (start / duration_secs) * width_px;
                let width = ((end - start).max(0.08) / duration_secs) * width_px;
                let mut lane_index = 0usize;
                while lane_index + 1 < lane_end_x.len() && lane_end_x[lane_index] > left {
                    lane_index += 1;
                }
                if lane_end_x[lane_index] > left && lane_index == lane_end_x.len() - 1 {
                    let min_lane = lane_end_x
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    lane_index = min_lane;
                }
                lane_end_x[lane_index] = left + width + 6.0;
                (left, width, lane_index)
            }
            _ => {
                let left = lane_end_x.iter().copied().fold(fallback_x, f64::max);
                fallback_x = left + 90.0;
                (left, 84.0, 2)
            }
        };
        let top = lane_tops[lane_index.min(lane_tops.len() - 1)];
        word_divs.push_str(&format!(
            "<div class=\"{class}\" style=\"left:{left:.1}px;top:{top:.1}px;width:{width:.1}px\" title=\"{title}: {text}\" data-full-word=\"{text}\">{text}</div>",
            text = html_escape(&word.text)
        ));
    }

    format!(
        "<section><div class=\"row-title\">{}</div><div class=\"transcript-line\">{}</div><div class=\"timeline\"><div class=\"track\">{}{}</div><div class=\"axis\"><span>0.00s</span><span>{:.2}s</span></div></div></section>",
        html_escape(title),
        transcript_line,
        boundaries,
        word_divs,
        duration_secs
    )
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
}

fn file_url_for_path(path: &Path) -> Result<String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::fs::canonicalize(path).with_context(|| format!("canonicalizing {}", path.display()))?
    };
    Ok(format!("file://{}", absolute.display()))
}
