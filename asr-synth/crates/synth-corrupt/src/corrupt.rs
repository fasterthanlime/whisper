use crate::cmudict::CmuDict;
use std::collections::HashMap;

/// Phoneme index for fast nearest-neighbor lookup.
///
/// Words are bucketed by (phoneme_count, first_phoneme) for fast filtering.
/// Two-word search uses split-point enumeration with pre-filtered candidates.
pub struct PhonemeIndex {
    /// (phoneme_count, first_phoneme) → Vec<(word, phonemes)>
    buckets: HashMap<(usize, String), Vec<(String, Vec<String>)>>,
    /// All entries for iteration
    all: Vec<(String, Vec<String>)>,
}

impl PhonemeIndex {
    pub fn new(dict: &CmuDict) -> Self {
        let mut buckets: HashMap<(usize, String), Vec<(String, Vec<String>)>> = HashMap::new();
        let mut all = Vec::with_capacity(dict.len());

        for (word, phonemes) in dict {
            let key = (phonemes.len(), phonemes.first().cloned().unwrap_or_default());
            buckets
                .entry(key)
                .or_default()
                .push((word.to_lowercase(), phonemes.clone()));
            all.push((word.to_lowercase(), phonemes.clone()));
        }

        Self { buckets, all }
    }

    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Find single words phonetically close to `target`.
    pub fn find_single_word(
        &self,
        target: &[String],
        max_dist: usize,
        max_results: usize,
    ) -> Vec<(String, usize)> {
        let target_len = target.len();
        let mut candidates: Vec<(String, usize)> = Vec::new();

        // Check buckets within ±max_dist of target length
        for delta in 0..=max_dist {
            for len in [target_len.wrapping_sub(delta), target_len + delta] {
                if len == 0 || len > 50 {
                    continue;
                }
                // Check all first-phoneme buckets at this length
                for ((blen, _), entries) in &self.buckets {
                    if *blen != len {
                        continue;
                    }
                    for (word, phonemes) in entries {
                        let dist = phoneme_edit_distance(target, phonemes);
                        if dist > 0 && dist <= max_dist {
                            candidates.push((word.clone(), dist));
                        }
                    }
                }
            }
        }

        candidates.sort_by_key(|(_, d)| *d);
        candidates.dedup_by(|a, b| a.0 == b.0);
        candidates.truncate(max_results);
        candidates
    }

    /// Find two-word phrases phonetically close to `target`.
    ///
    /// For each split point in the target phoneme sequence, find the best
    /// left and right words independently, then combine.
    pub fn find_two_word(
        &self,
        target: &[String],
        max_dist: usize,
        max_results: usize,
    ) -> Vec<(String, usize)> {
        let target_len = target.len();
        if target_len < 2 {
            return Vec::new();
        }

        let mut candidates: Vec<(String, usize)> = Vec::new();

        for split in 1..target_len {
            let left_target = &target[..split];
            let right_target = &target[split..];

            // Find best left matches (only short words)
            let left_matches = self.find_half(left_target, max_dist);
            let right_matches = self.find_half(right_target, max_dist);

            // Combine: total distance ≤ max_dist
            for (lw, ld) in &left_matches {
                for (rw, rd) in &right_matches {
                    let total = ld + rd;
                    if total > 0 && total <= max_dist {
                        candidates.push((format!("{} {}", lw, rw), total));
                    }
                }
            }
        }

        candidates.sort_by_key(|(_, d)| *d);
        candidates.dedup_by(|a, b| a.0 == b.0);
        candidates.truncate(max_results);
        candidates
    }

    /// Find words close to a phoneme subsequence (used for split matching).
    fn find_half(&self, target: &[String], max_dist: usize) -> Vec<(String, usize)> {
        let target_len = target.len();
        let mut results: Vec<(String, usize)> = Vec::new();

        // Only check words within ±1 phoneme of target length
        for ((_blen, _), entries) in &self.buckets {
            if (*_blen as isize - target_len as isize).unsigned_abs() > 1 {
                continue;
            }
            for (word, phonemes) in entries {
                let dist = phoneme_edit_distance(target, phonemes);
                if dist <= max_dist {
                    results.push((word.clone(), dist));
                }
            }
        }

        results.sort_by_key(|(_, d)| *d);
        results.truncate(10); // Keep top 10 per half
        results
    }
}

/// Weighted phoneme edit distance.
///
/// Uses phoneme feature similarity for substitution costs:
/// K→G (voicing flip) costs ~0.2, K→B (place change) costs ~0.8.
/// Insertion/deletion cost 1.0.
/// Returns distance * 100 as integer for easy comparison.
pub fn phoneme_edit_distance(a: &[String], b: &[String]) -> usize {
    use crate::features::substitution_cost;

    let m = a.len();
    let n = b.len();
    let mut dp = vec![0u32; (m + 1) * (n + 1)];

    for i in 0..=m {
        dp[i * (n + 1)] = (i as u32) * 100;
    }
    for j in 0..=n {
        dp[j] = (j as u32) * 100;
    }
    for i in 1..=m {
        for j in 1..=n {
            let sub_cost = (substitution_cost(&a[i - 1], &b[j - 1]) * 100.0) as u32;
            dp[i * (n + 1) + j] = (dp[(i - 1) * (n + 1) + j] + 100)        // deletion
                .min(dp[i * (n + 1) + (j - 1)] + 100)                        // insertion
                .min(dp[(i - 1) * (n + 1) + (j - 1)] + sub_cost);            // substitution
        }
    }
    dp[m * (n + 1) + n] as usize
}
