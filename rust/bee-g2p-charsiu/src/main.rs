use bee_g2p_charsiu::{
    CharsiuSidecarClient, PhonemizeTextRequest, PhonemizeWordsRequest, ProbeRequest,
    probe_text_default, summarize_probe_runs, token_piece_comparison_tokens, token_piece_ipa_spans,
    token_piece_phones,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut lang_code = "eng-us".to_string();
    let mut text = None;
    let mut probe_text = None;
    let mut token_spans_text = None;
    let mut token_phones_text = None;
    let mut comparison_tokens_text = None;
    let mut words = Vec::new();

    while let Some(arg) = args.next() {
        if arg == "--lang-code" {
            let value = args.next().ok_or("--lang-code requires a value")?;
            lang_code = value;
            continue;
        }
        if arg == "--text" {
            let value = args.next().ok_or("--text requires a value")?;
            text = Some(value);
            continue;
        }
        if arg == "--probe-text" {
            let value = args.next().ok_or("--probe-text requires a value")?;
            probe_text = Some(value);
            continue;
        }
        if arg == "--token-spans-text" {
            let value = args.next().ok_or("--token-spans-text requires a value")?;
            token_spans_text = Some(value);
            continue;
        }
        if arg == "--token-phones-text" {
            let value = args.next().ok_or("--token-phones-text requires a value")?;
            token_phones_text = Some(value);
            continue;
        }
        if arg == "--comparison-tokens-text" {
            let value = args
                .next()
                .ok_or("--comparison-tokens-text requires a value")?;
            comparison_tokens_text = Some(value);
            continue;
        }
        words.push(arg);
    }

    let selected_modes = usize::from(text.is_some())
        + usize::from(probe_text.is_some())
        + usize::from(token_spans_text.is_some())
        + usize::from(token_phones_text.is_some())
        + usize::from(comparison_tokens_text.is_some());
    if selected_modes > 1
        || ((text.is_some()
            || probe_text.is_some()
            || token_spans_text.is_some()
            || token_phones_text.is_some()
            || comparison_tokens_text.is_some())
            && !words.is_empty())
    {
        return Err(
            "use exactly one of --text TEXT, --probe-text TEXT, --token-spans-text TEXT, --token-phones-text TEXT, --comparison-tokens-text TEXT, or WORD..."
                .into(),
        );
    }

    if text.is_none()
        && probe_text.is_none()
        && token_spans_text.is_none()
        && token_phones_text.is_none()
        && comparison_tokens_text.is_none()
        && words.is_empty()
    {
        return Err(
            "usage: bee-g2p-charsiu [--lang-code CODE] [--text TEXT | --probe-text TEXT | --token-spans-text TEXT | --token-phones-text TEXT | --comparison-tokens-text TEXT | WORD...]"
                .into()
        );
    }

    if let Some(text) = comparison_tokens_text {
        let result = probe_text_default(ProbeRequest {
            text,
            lang_code,
            top_k: 6,
        })?;
        println!("text\t{}", result.text);
        println!("decoded_ipa\t{}", result.decoded_output);
        for token in token_piece_comparison_tokens(&result) {
            let word = token.word_surface.unwrap_or_default();
            println!(
                "cmp\t{}\tword={}\ttoken={}\tphone={}\tspan={}..{}",
                token.token_surface,
                word,
                token.token,
                token.comparison_token,
                token.ipa_source_start,
                token.ipa_source_end
            );
        }
        return Ok(());
    }

    if let Some(text) = token_phones_text {
        let result = probe_text_default(ProbeRequest {
            text,
            lang_code,
            top_k: 6,
        })?;
        println!("text\t{}", result.text);
        println!("decoded_ipa\t{}", result.decoded_output);
        for span in token_piece_phones(&result) {
            let word = span.word_surface.unwrap_or_default();
            println!(
                "phones\t{}\tword={}\ttoken={}\traw={}\tnorm={}",
                span.token_surface,
                word,
                span.token,
                span.ipa_tokens.join(" "),
                span.normalized_phones.join(" ")
            );
        }
        return Ok(());
    }

    if let Some(text) = token_spans_text {
        let result = probe_text_default(ProbeRequest {
            text,
            lang_code,
            top_k: 6,
        })?;
        println!("text\t{}", result.text);
        println!("decoded_ipa\t{}", result.decoded_output);
        for span in token_piece_ipa_spans(&result) {
            let word = span.word_surface.unwrap_or_default();
            println!(
                "span\t{}..{}\t{}\tword={}\ttoken={}\tsurface={}",
                span.ipa_step_start,
                span.ipa_step_end,
                span.ipa_text,
                word,
                span.token,
                span.token_surface
            );
        }
        return Ok(());
    }

    if let Some(text) = probe_text {
        let result = probe_text_default(ProbeRequest {
            text,
            lang_code,
            top_k: 6,
        })?;
        println!("text\t{}", result.text);
        println!("decoded_ipa\t{}", result.decoded_output);
        for run in summarize_probe_runs(&result) {
            let word = run.word_surface.unwrap_or_default();
            println!(
                "run\t{}..{}\t{}\tword={}\tqwen={}",
                run.output_start, run.output_end, run.rendered_output, word, run.qwen_piece_token
            );
        }
        return Ok(());
    }

    let mut client = CharsiuSidecarClient::spawn_default()?;
    eprintln!(
        "charsiu ready model={} device={}",
        client.ready().model,
        client.ready().device
    );
    if let Some(text) = text {
        let result = client.phonemize_text(PhonemizeTextRequest { text, lang_code })?;
        for row in result.word_ipas {
            println!(
                "{}..{}\t{}\t{}",
                row.char_start, row.char_end, row.word, row.ipa
            );
        }
    } else {
        let result = client.phonemize_words(PhonemizeWordsRequest { words, lang_code })?;
        for row in result.word_ipas {
            println!("{}\t{}", row.word, row.ipa);
        }
    }
    Ok(())
}
