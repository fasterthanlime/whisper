use rand::prelude::*;
use std::collections::HashMap;

/// Simple word-level markov chain text generator.
pub struct MarkovChain {
    /// (word_a, word_b) → vec of possible next words
    transitions: HashMap<(String, String), Vec<String>>,
    /// All bigram starts that begin sentences (after a period or start of text)
    starters: Vec<(String, String)>,
}

impl MarkovChain {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            starters: Vec::new(),
        }
    }

    /// Feed a text into the chain.
    pub fn feed(&mut self, text: &str) {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 3 {
            return;
        }

        // First bigram is a sentence starter
        self.starters
            .push((words[0].to_string(), words[1].to_string()));

        for window in words.windows(3) {
            let key = (window[0].to_string(), window[1].to_string());
            self.transitions
                .entry(key)
                .or_default()
                .push(window[2].to_string());

            // If word ends with sentence-ending punctuation, next bigram is a starter
            if window[1].ends_with('.') || window[1].ends_with('!') || window[1].ends_with('?') {
                if window.len() > 2 {
                    // The next word starts a new sentence
                    // (handled by the next window iteration)
                }
            }
        }
    }

    /// Generate a sentence of approximately `target_words` words.
    pub fn generate(&self, rng: &mut impl Rng, target_words: usize) -> Option<String> {
        if self.starters.is_empty() {
            return None;
        }

        let (mut w1, mut w2) = self.starters.choose(rng)?.clone();
        let mut result = vec![w1.clone(), w2.clone()];

        for _ in 0..target_words + 10 {
            let key = (w1.clone(), w2.clone());
            let next = match self.transitions.get(&key) {
                Some(options) => options.choose(rng)?.clone(),
                None => break,
            };
            result.push(next.clone());
            w1 = w2;
            w2 = next;

            // Stop at sentence boundary if we've hit target length
            if result.len() >= target_words {
                let last = result.last().unwrap();
                if last.ends_with('.') || last.ends_with('!') || last.ends_with('?') {
                    break;
                }
            }

            // Hard cap
            if result.len() > target_words + 20 {
                break;
            }
        }

        let text = result.join(" ");
        if text.len() < 10 {
            return None;
        }
        Some(text)
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}
