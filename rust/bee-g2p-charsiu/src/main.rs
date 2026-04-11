use bee_g2p_charsiu::{
    CharsiuSidecarClient, PhonemizeTextRequest, PhonemizeWordsRequest, ProbeRequest,
    probe_text_default,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut lang_code = "eng-us".to_string();
    let mut text = None;
    let mut probe_text = None;
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
        words.push(arg);
    }

    let selected_modes = usize::from(text.is_some()) + usize::from(probe_text.is_some());
    if selected_modes > 1 || ((text.is_some() || probe_text.is_some()) && !words.is_empty()) {
        return Err("use exactly one of --text TEXT, --probe-text TEXT, or WORD...".into());
    }

    if text.is_none() && probe_text.is_none() && words.is_empty() {
        return Err(
            "usage: bee-g2p-charsiu [--lang-code CODE] [--text TEXT | --probe-text TEXT | WORD...]"
                .into(),
        );
    }

    if let Some(text) = probe_text {
        let result = probe_text_default(ProbeRequest {
            text,
            lang_code,
            top_k: 6,
        })?;
        println!("text\t{}", result.text);
        println!("decoded_ipa\t{}", result.decoded_output);
        for row in result.cross_attention {
            if row.output_piece.is_empty() || row.output_piece == "</s>" {
                continue;
            }
            let top_word = row.top_word_surface.unwrap_or_default();
            let top_qwen = row.top_qwen_piece_token.unwrap_or_default();
            println!(
                "out[{}]\t{}\tword={}\tqwen={}",
                row.output_index, row.output_piece, top_word, top_qwen
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
