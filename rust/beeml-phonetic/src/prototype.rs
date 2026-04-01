pub fn parse_reviewed_ipa(ipa_text: &str) -> Vec<String> {
    ipa_text
        .split_whitespace()
        .map(std::string::ToString::to_string)
        .collect()
}

pub fn phoneme_similarity(a: &[String], b: &[String]) -> Option<f32> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let distance = levenshtein(a, b) as f32;
    let max_len = a.len().max(b.len()) as f32;
    let normalized = 1.0 - (distance / max_len);
    Some(normalized.clamp(0.0, 1.0))
}

fn levenshtein(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for (i, ax) in a.iter().enumerate() {
        curr[0] = i + 1;
        let mut prev_left = i;
        for (j, by) in b.iter().enumerate() {
            let cost = usize::from(ax != by);
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev_left + cost);
            prev_left = prev[j + 1];
            prev[j + 1] = curr[j + 1];
        }
        prev.copy_from_slice(&curr);
    }

    prev[b.len()]
}
