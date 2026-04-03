use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use bee_phonetic::{
    enumerate_transcript_spans_with, parse_reviewed_ipa, query_index, verify_shortlist,
    PhoneticIndex, RetrievalQuery, SeedDataset, TranscriptAlignmentToken,
};
use bee_transcribe::{Engine, EngineConfig, SessionOptions};
use beeml::rpc::{
    AcceptedEdit, AliasSource, BeeMl, CorrectionDebugResult, CorrectionRequest, CorrectionResult,
    FilterDecision, IdentifierFlags, RerankerDebugTrace, RetrievalCandidateDebug,
    RetrievalIndexView, RetrievalLaneHit, RetrievalPrototypeEvalRequest,
    RetrievalPrototypeEvalResult, RetrievalPrototypeProbeRequest, RetrievalPrototypeProbeResult,
    SpanDebugTrace, SpanDebugView, TermAliasView, TermInspectionRequest, TermInspectionResult,
    TimingBreakdown, TranscribeWavResult,
};
use tokio::net::TcpListener;
use vox::{NoopClient, Rx, Tx};

#[derive(Clone)]
struct BeeMlService {
    inner: Arc<BeemlServiceInner>,
}

struct BeemlServiceInner {
    engine: Engine,
    index: PhoneticIndex,
}

impl BeeMl for BeeMlService {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String> {
        let samples = bee_transcribe::decode_wav(&wav_bytes).map_err(|e| e.to_string())?;

        let mut session = self.inner.engine.session(SessionOptions::default());

        session.feed(&samples).map_err(|e| e.to_string())?;
        let update = session.finish().map_err(|e| e.to_string())?;

        Ok(TranscribeWavResult {
            transcript: update.text,
            words: update.alignments,
        })
    }

    async fn stream_transcribe(
        &self,
        _audio_in: Rx<Vec<f32>>,
        _updates_out: Tx<bee_transcribe::Update>,
    ) -> Result<(), String> {
        Err("stream_transcribe is temporarily unavailable while beeml migrates to the new correction RPC surface".to_string())
    }

    async fn correct_transcript(
        &self,
        request: CorrectionRequest,
    ) -> Result<CorrectionResult, String> {
        Ok(CorrectionResult {
            original_transcript: request.transcript.clone(),
            corrected_transcript: request.transcript,
            accepted_edits: Vec::<AcceptedEdit>::new(),
        })
    }

    async fn debug_correction(
        &self,
        request: CorrectionRequest,
    ) -> Result<CorrectionDebugResult, String> {
        Ok(CorrectionDebugResult {
            result: CorrectionResult {
                original_transcript: request.transcript.clone(),
                corrected_transcript: request.transcript,
                accepted_edits: Vec::new(),
            },
            spans: Vec::new(),
            reranker_regions: Vec::<RerankerDebugTrace>::new(),
            timings: TimingBreakdown {
                span_enumeration_ms: 0,
                retrieval_ms: 0,
                verify_ms: 0,
                rerank_ms: 0,
                total_ms: 0,
            },
        })
    }

    async fn probe_retrieval_prototype(
        &self,
        request: RetrievalPrototypeProbeRequest,
    ) -> Result<RetrievalPrototypeProbeResult, String> {
        let alignments = if request.words.is_empty() {
            None
        } else {
            Some(
                request
                    .words
                    .iter()
                    .map(|word| TranscriptAlignmentToken {
                        start_time: word.start,
                        end_time: word.end,
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let mut g2p = EspeakG2p::default();
        let spans = enumerate_transcript_spans_with(
            &request.transcript,
            request.max_span_words as usize,
            alignments.as_deref(),
            |text| g2p.ipa_tokens(text).ok().flatten(),
        );

        let traces = spans
            .into_iter()
            .map(|span| {
                let shortlist = query_index(
                    &self.inner.index,
                    &RetrievalQuery {
                        text: span.text.clone(),
                        ipa_tokens: span.ipa_tokens.clone(),
                        reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                        token_count: (span.token_end - span.token_start) as u8,
                    },
                    request.shortlist_limit as usize,
                );
                let verified = verify_shortlist(
                    &span,
                    &shortlist,
                    &self.inner.index,
                    request.verify_limit as usize,
                );
                let verified_by_alias = verified
                    .into_iter()
                    .map(|candidate| (candidate.alias_id, candidate))
                    .collect::<HashMap<_, _>>();

                let candidates = shortlist
                    .into_iter()
                    .map(|candidate| {
                        let verified = verified_by_alias.get(&candidate.alias_id);
                        RetrievalCandidateDebug {
                            alias_id: candidate.alias_id,
                            term: candidate.term,
                            alias_text: candidate.alias_text,
                            alias_source: map_alias_source(candidate.alias_source),
                            lane_hits: vec![RetrievalLaneHit {
                                view: map_index_view(candidate.matched_view),
                                qgram_overlap: candidate.qgram_overlap,
                                boundary_overlap: 0,
                            }],
                            coarse_score: candidate.coarse_score,
                            best_view_score: candidate.best_view_score,
                            cross_view_support: candidate.cross_view_support,
                            token_bonus: candidate.token_bonus,
                            phone_bonus: candidate.phone_bonus,
                            extra_length_penalty: candidate.extra_length_penalty,
                            phonetic_score: verified.map(|v| v.phonetic_score).unwrap_or(0.0),
                            token_count_match: candidate.token_count_match,
                            phone_count_delta: candidate.phone_count_delta,
                            total_qgram_overlap: candidate.total_qgram_overlap,
                            filter_decisions: vec![
                                FilterDecision {
                                    name: "token_count".to_string(),
                                    passed: candidate.token_count_match,
                                    detail: format!(
                                        "token_count_match={}",
                                        candidate.token_count_match
                                    ),
                                },
                                FilterDecision {
                                    name: "phone_count".to_string(),
                                    passed: candidate.phone_count_delta.abs() <= 3,
                                    detail: format!(
                                        "phone_count_delta={}",
                                        candidate.phone_count_delta
                                    ),
                                },
                                FilterDecision {
                                    name: "verified".to_string(),
                                    passed: verified.is_some(),
                                    detail: verified
                                        .map(|v| format!("phonetic_score={:.3}", v.phonetic_score))
                                        .unwrap_or_else(|| "not_in_verified_shortlist".to_string()),
                                },
                            ],
                            reached_reranker: false,
                            accepted: false,
                        }
                    })
                    .collect();

                SpanDebugTrace {
                    span: SpanDebugView {
                        token_start: span.token_start as u32,
                        token_end: span.token_end as u32,
                        char_start: span.char_start as u32,
                        char_end: span.char_end as u32,
                        start_sec: span.start_sec.unwrap_or(0.0),
                        end_sec: span.end_sec.unwrap_or(0.0),
                        text: span.text,
                        ipa_tokens: span.ipa_tokens,
                        reduced_ipa_tokens: span.reduced_ipa_tokens,
                        feature_tokens: Vec::new(),
                    },
                    candidates,
                }
            })
            .collect();

        Ok(RetrievalPrototypeProbeResult {
            transcript: request.transcript,
            spans: traces,
            timings: TimingBreakdown {
                span_enumeration_ms: 0,
                retrieval_ms: 0,
                verify_ms: 0,
                rerank_ms: 0,
                total_ms: 0,
            },
        })
    }

    async fn inspect_term(
        &self,
        request: TermInspectionRequest,
    ) -> Result<TermInspectionResult, String> {
        let aliases = self
            .inner
            .index
            .aliases
            .iter()
            .filter(|alias| alias.term.eq_ignore_ascii_case(&request.term))
            .map(|alias| TermAliasView {
                alias_text: alias.alias_text.clone(),
                alias_source: map_alias_source(alias.alias_source),
                ipa_tokens: alias.ipa_tokens.clone(),
                reduced_ipa_tokens: alias.reduced_ipa_tokens.clone(),
                feature_tokens: Vec::new(),
                identifier_flags: map_identifier_flags(&alias.identifier_flags),
            })
            .collect();

        Ok(TermInspectionResult {
            term: request.term,
            aliases,
        })
    }

    async fn run_retrieval_prototype_eval(
        &self,
        _request: RetrievalPrototypeEvalRequest,
    ) -> Result<RetrievalPrototypeEvalResult, String> {
        Err("run_retrieval_prototype_eval is not implemented yet".to_string())
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let listen_addr = env::var("BML_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:9944".to_string());
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set")?;
    let tokenizer_path = env::var("BEE_TOKENIZER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir.join("tokenizer.json"));
    let aligner_dir = env::var("BEE_ALIGNER_DIR")
        .map(PathBuf::from)
        .context("BEE_ALIGNER_DIR must be set")?;

    eprintln!("loading ASR engine from {}", model_dir.display());
    let engine = Engine::load(&EngineConfig {
        model_dir: &model_dir,
        tokenizer_path: &tokenizer_path,
        aligner_dir: &aligner_dir,
    })
    .context("loading engine")?;

    let dataset =
        SeedDataset::load_canonical().context("loading canonical phonetic seed dataset")?;
    dataset
        .validate()
        .context("validating canonical phonetic seed dataset")?;
    let index = dataset.phonetic_index();

    let handler = BeeMlService {
        inner: Arc::new(BeemlServiceInner { engine, index }),
    };

    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("binding websocket listener on {listen_addr}"))?;

    eprintln!("beeml vox websocket server listening on ws://{listen_addr}");

    loop {
        let (stream, peer_addr) = listener
            .accept()
            .await
            .context("accepting websocket socket")?;
        let handler = handler.clone();

        tokio::spawn(async move {
            let link = match vox_websocket::WsLink::server(stream).await {
                Ok(link) => link,
                Err(error) => {
                    eprintln!("websocket handshake failed for {peer_addr}: {error}");
                    return;
                }
            };

            let establish = vox_core::acceptor_on(link)
                .on_connection(beeml::rpc::BeeMlDispatcher::new(handler))
                .establish::<NoopClient>()
                .await;

            match establish {
                Ok(client) => {
                    eprintln!("client connected: {peer_addr}");
                    client.caller.closed().await;
                    eprintln!("client disconnected: {peer_addr}");
                }
                Err(error) => {
                    eprintln!("vox session establish failed for {peer_addr}: {error}");
                }
            }
        });
    }
}

fn map_alias_source(source: bee_phonetic::AliasSource) -> AliasSource {
    match source {
        bee_phonetic::AliasSource::Canonical => AliasSource::Canonical,
        bee_phonetic::AliasSource::Spoken => AliasSource::Spoken,
        bee_phonetic::AliasSource::Confusion => AliasSource::Confusion,
    }
}

fn map_index_view(view: bee_phonetic::IndexView) -> RetrievalIndexView {
    match view {
        bee_phonetic::IndexView::RawIpa2 => RetrievalIndexView::RawIpa2,
        bee_phonetic::IndexView::RawIpa3 => RetrievalIndexView::RawIpa3,
        bee_phonetic::IndexView::ReducedIpa2 => RetrievalIndexView::ReducedIpa2,
        bee_phonetic::IndexView::ReducedIpa3 => RetrievalIndexView::ReducedIpa3,
    }
}

fn map_identifier_flags(flags: &bee_phonetic::IdentifierFlags) -> IdentifierFlags {
    IdentifierFlags {
        acronym_like: flags.acronym_like,
        has_digits: flags.has_digits,
        snake_like: flags.snake_like,
        camel_like: flags.camel_like,
        symbol_like: flags.symbol_like,
    }
}

#[derive(Default)]
struct EspeakG2p {
    cache: HashMap<String, Vec<String>>,
}

impl EspeakG2p {
    fn ipa_tokens(&mut self, text: &str) -> Result<Option<Vec<String>>, String> {
        let key = text.trim();
        if key.is_empty() {
            return Ok(None);
        }
        if let Some(tokens) = self.cache.get(key) {
            return Ok(Some(tokens.clone()));
        }

        let output = Command::new("espeak-ng")
            .args(["-q", "--ipa=3", "--sep= "])
            .arg(key)
            .output()
            .map_err(|e| format!("spawn espeak-ng for '{key}': {e}"))?;
        if !output.status.success() {
            return Err(format!("espeak-ng failed for '{key}'"));
        }

        let ipa = String::from_utf8(output.stdout)
            .map_err(|e| format!("decode espeak-ng output for '{key}': {e}"))?;
        let tokens = parse_reviewed_ipa(ipa.trim());
        if tokens.is_empty() {
            return Ok(None);
        }

        self.cache.insert(key.to_string(), tokens.clone());
        Ok(Some(tokens))
    }
}
