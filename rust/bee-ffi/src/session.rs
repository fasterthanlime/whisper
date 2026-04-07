/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_engine_get_stats(engine: *const AsrEngine) -> AsrEngineStats {
    if engine.is_null() {
        return AsrEngineStats::default();
    }
    unsafe { &*engine }.stats.get()
}

/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_create(
    engine: *const AsrEngine,
    opts: AsrSessionOptions,
) -> *mut AsrSession {
    if engine.is_null() {
        return std::ptr::null_mut();
    }

    let engine_ref = unsafe { &*engine };
    let arc = engine_ref.inner.clone();

    let chunk_duration = if opts.chunk_size_sec > 0.0 {
        opts.chunk_size_sec
    } else {
        0.4
    };
    let max_streaming = if opts.max_new_tokens_streaming > 0 {
        opts.max_new_tokens_streaming as usize
    } else {
        32
    };
    let max_final = if opts.max_new_tokens_final > 0 {
        opts.max_new_tokens_final as usize
    } else {
        512
    };

    let language = if !opts.language.is_null() {
        let s = unsafe { CStr::from_ptr(opts.language) }
            .to_str()
            .unwrap_or("");
        if s.is_empty() {
            Language::default()
        } else {
            Language(s.to_string())
        }
    } else {
        Language::default()
    };

    let commit_token_count = if opts.unfixed_token_num > 0 {
        opts.unfixed_token_num as usize
    } else {
        12
    };
    let rollback_tokens = if opts.rollback_token_num > 0 {
        opts.rollback_token_num as usize
    } else {
        5
    };

    let session_opts = SessionOptions {
        chunk_duration,
        commit_token_count,
        rollback_tokens,
        max_tokens_streaming: max_streaming,
        max_tokens_final: max_final,
        language,
        ..SessionOptions::default()
    };

    // SAFETY: The Arc keeps the engine alive for the lifetime of the session.
    // We transmute the lifetime to 'static since FFI can't express borrows.
    let engine_ref: &Engine = &arc;
    let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
    let mut session = engine_static.session(session_opts);

    // Create VAD from pre-loaded tensors
    let asr_engine = unsafe { &*engine };
    if let Some(ref tensors) = asr_engine.vad_tensors {
        match bee_vad::SileroVad::from_tensors(tensors) {
            Ok(vad) => session.set_vad(vad),
            Err(e) => ffi_log(&format!("Failed to create VAD: {e}")),
        }
    }

    Box::into_raw(Box::new(AsrSession {
        _engine: arc,
        session,
        last_text: String::new(),
        finished: false,
    }))
}

#[repr(C)]
pub struct AsrFeedResult {
    pub text: *mut c_char,
    pub committed_utf16_len: usize,
    pub alignments_json: *mut c_char,
    pub debug_json: *mut c_char,
}

impl AsrFeedResult {
    fn empty() -> Self {
        Self {
            text: std::ptr::null_mut(),
            committed_utf16_len: 0,
            alignments_json: std::ptr::null_mut(),
            debug_json: std::ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_session_feed(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    feed_impl(session, samples, num_samples, out_err)
}

#[no_mangle]
pub extern "C" fn asr_session_feed_finalizing(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    // With the new API, there's no separate "finalizing" feed — just feed
    // normally and call finish() when done. For backward compat, treat this
    // the same as a regular feed.
    feed_impl(session, samples, num_samples, out_err)
}

fn feed_impl(
    session: *mut AsrSession,
    samples: *const c_float,
    num_samples: usize,
    out_err: *mut *mut c_char,
) -> AsrFeedResult {
    if session.is_null() || samples.is_null() {
        return AsrFeedResult::empty();
    }

    let session = unsafe { &mut *session };
    if session.finished {
        return AsrFeedResult::empty();
    }
    let audio = unsafe { std::slice::from_raw_parts(samples, num_samples) };

    match session.session.feed(audio) {
        Ok(Some(update)) => {
            if update.text != session.last_text {
                ffi_log(&format!(
                    "FEED text changed:\n  was: {:?}\n  now: {:?}",
                    session.last_text, update.text
                ));
            }
            session.last_text = update.text.clone();

            let committed_utf16_len = update.text[..update.committed_len].encode_utf16().count();

            let alignments_json = if update.alignments.is_empty() {
                std::ptr::null_mut()
            } else {
                let json = serde_json::to_string(
                    &update
                        .alignments
                        .iter()
                        .map(|a| {
                            serde_json::json!({
                                "word": a.word,
                                "start": (a.start * 1000.0).round() / 1000.0,
                                "end": (a.end * 1000.0).round() / 1000.0,
                            })
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap_or_else(|_| "[]".to_string());
                to_c_string(&json)
            };

            AsrFeedResult {
                text: to_c_string(&update.text),
                committed_utf16_len,
                alignments_json,
                debug_json: std::ptr::null_mut(),
            }
        }
        Ok(None) => AsrFeedResult::empty(),
        Err(e) => {
            ffi_log(&format!("FEED => error: {e}"));
            set_err(out_err, &e.to_string());
            AsrFeedResult::empty()
        }
    }
}

#[no_mangle]
pub extern "C" fn asr_feed_result_free(result: AsrFeedResult) {
    if !result.text.is_null() {
        unsafe { drop(CString::from_raw(result.text)) };
    }
    if !result.alignments_json.is_null() {
        unsafe { drop(CString::from_raw(result.alignments_json)) };
    }
    if !result.debug_json.is_null() {
        unsafe { drop(CString::from_raw(result.debug_json)) };
    }
}

/// # Safety
/// `session` must be a valid session pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_finish(
    session: *mut AsrSession,
    out_err: *mut *mut c_char,
) -> *mut c_char {
    if session.is_null() {
        return std::ptr::null_mut();
    }

    // Take ownership of the session to call finish() which consumes it.
    // We reconstruct a new dummy session in its place.
    let session = unsafe { &mut *session };

    if session.finished {
        return std::ptr::null_mut();
    }
    session.finished = true;

    // We can't consume session.session through a pointer, so we use a
    // temporary: swap in a fresh session, finish the real one.
    let arc = session._engine.clone();
    let engine_ref: &Engine = &arc;
    let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
    let real_session = std::mem::replace(
        &mut session.session,
        engine_static.session(SessionOptions::default()),
    );

    match real_session.finish() {
        Ok(update) => {
            session.last_text = update.text.clone();
            to_c_string(&update.text)
        }
        Err(e) => {
            set_err(out_err, &e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `session` must be a valid session pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_last_language(_session: *const AsrSession) -> *mut c_char {
    // Language is set at session creation and doesn't change.
    to_c_string("English")
}

#[no_mangle]
pub extern "C" fn asr_session_set_language(
    _session: *mut AsrSession,
    _language: *const c_char,
    _out_err: *mut *mut c_char,
) -> bool {
    true
}

/// # Safety
/// `session` must be a valid session pointer.
#[no_mangle]
pub unsafe extern "C" fn asr_session_free(session: *mut AsrSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}

/// # Safety
/// `s` must be either null or a pointer returned by this module.
#[no_mangle]
pub unsafe extern "C" fn asr_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe { drop(CString::from_raw(s)) };
    }
}
