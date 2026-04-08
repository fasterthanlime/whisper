//! Sparse FTRL-Proximal optimizer for online linear classification.
//!
//! Only stores weights for features that have received non-zero updates,
//! making it efficient for high-dimensional hashed feature spaces.
//!
//! Reference: McMahan et al., "Ad Click Prediction: a View from the Trenches" (2013)

use std::collections::HashMap;
use std::io::{Read, Write};

/// Per-feature accumulators: (z, n) where z is the lazy weight accumulator
/// and n is the sum of squared gradients.
#[derive(Clone, Debug)]
pub struct SparseFtrl {
    /// (z_i, n_i) accumulators, keyed by feature index.
    accumulators: HashMap<u64, (f64, f64)>,
    /// Learning rate parameter.
    pub alpha: f64,
    /// Learning rate smoothing parameter.
    pub beta: f64,
    /// L1 regularization (sparsity).
    pub l1: f64,
    /// L2 regularization.
    pub l2: f64,
    /// Feature indices that are frozen (not updated during training).
    frozen: std::collections::HashSet<u64>,
    /// Pre-computed frozen weights loaded from file. Checked first in weight_for().
    loaded_weights: HashMap<u64, f64>,
}

/// A single active feature: index + value.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub index: u64,
    pub value: f64,
}

impl SparseFtrl {
    /// Create a new FTRL optimizer.
    ///
    /// - `alpha`: learning rate (higher = faster adaptation, more noise). Typical: 0.1–1.0.
    /// - `beta`: learning rate smoothing (higher = more conservative early updates). Typical: 1.0.
    /// - `l1`: L1 regularization (drives unused weights to zero → sparsity). Typical: 0.0001.
    /// - `l2`: L2 regularization (prevents large weights). Typical: 0.001.
    pub fn new(alpha: f64, beta: f64, l1: f64, l2: f64) -> Self {
        Self {
            accumulators: HashMap::new(),
            alpha,
            beta,
            l1,
            l2,
            frozen: std::collections::HashSet::new(),
            loaded_weights: HashMap::new(),
        }
    }

    /// Freeze a set of feature indices — they will not be updated during training.
    pub fn freeze(&mut self, indices: impl IntoIterator<Item = u64>) {
        self.frozen.extend(indices);
    }

    /// Compute the weight for a single feature from its (z, n) accumulators.
    ///
    /// Applies L1 proximal thresholding: if |z| ≤ l1, the weight is zero
    /// (soft-thresholding for sparsity). Otherwise, w = -(z - sign(z)·l1) / (l2 + (β + √n) / α).
    fn weight_for(&self, index: u64) -> f64 {
        if let Some(&w) = self.loaded_weights.get(&index) {
            return w;
        }
        let Some(&(z, n)) = self.accumulators.get(&index) else {
            return 0.0;
        };
        if z.abs() <= self.l1 {
            0.0
        } else {
            let sign = if z > 0.0 { 1.0 } else { -1.0 };
            -(z - sign * self.l1) / ((self.beta + n.sqrt()) / self.alpha + self.l2)
        }
    }

    /// Predict the raw score (logit) for a set of active features.
    pub fn predict(&self, features: &[Feature]) -> f64 {
        features
            .iter()
            .map(|f| self.weight_for(f.index) * f.value)
            .sum()
    }

    /// Predict probability via sigmoid.
    pub fn predict_prob(&self, features: &[Feature]) -> f64 {
        sigmoid(self.predict(features))
    }

    /// Update weights given a set of active features and the true label.
    pub fn update(&mut self, features: &[Feature], label: bool) {
        let p = self.predict_prob(features);
        // Gradient of log-loss: g = p - y
        let g = p - if label { 1.0 } else { 0.0 };

        for f in features {
            if self.frozen.contains(&f.index) { continue; }
            let g_i = g * f.value;
            // Compute current weight before mutating accumulators
            let (z_old, n_old) = self.accumulators.get(&f.index).copied().unwrap_or((0.0, 0.0));
            let w_i = if z_old.abs() <= self.l1 {
                0.0
            } else {
                let sign = if z_old > 0.0 { 1.0 } else { -1.0 };
                -(z_old - sign * self.l1) / ((self.beta + n_old.sqrt()) / self.alpha + self.l2)
            };
            let sigma = ((n_old + g_i * g_i).sqrt() - n_old.sqrt()) / self.alpha;
            let z_new = z_old + g_i - sigma * w_i;
            let n_new = n_old + g_i * g_i;
            self.accumulators.insert(f.index, (z_new, n_new));
        }
    }

    /// Number of features with non-zero accumulators.
    pub fn num_active(&self) -> usize {
        self.accumulators.len()
    }

    /// Get all non-zero weights as (index, weight) pairs, sorted by index.
    pub fn weights(&self) -> Vec<(u64, f64)> {
        let mut w: Vec<(u64, f64)> = self
            .accumulators
            .keys()
            .map(|&idx| (idx, self.weight_for(idx)))
            .filter(|(_, w)| *w != 0.0)
            .collect();
        w.sort_by_key(|(idx, _)| *idx);
        w
    }

    /// Get the weight for a specific dense feature index.
    pub fn weight_at(&self, index: u64) -> f64 {
        self.weight_for(index)
    }

    /// Save all trained weights to a binary file (including L1-zeroed features).
    /// Format: u32 LE count, then (u64 LE index, f64 LE weight) pairs.
    /// Includes zero-weight entries so loaded models know those features should be zero,
    /// rather than falling through to accumulator-derived weights from seeding.
    pub fn save_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        let mut w: Vec<(u64, f64)> = self
            .accumulators
            .keys()
            .map(|&idx| (idx, self.weight_for(idx)))
            .collect();
        w.sort_by_key(|(idx, _)| *idx);
        writer.write_all(&(w.len() as u32).to_le_bytes())?;
        for (idx, val) in &w {
            writer.write_all(&idx.to_le_bytes())?;
            writer.write_all(&val.to_le_bytes())?;
        }
        Ok(())
    }

    /// Load frozen weights from a binary file, fully replacing the model.
    /// Clears all accumulators so seed weights don't leak through for features
    /// not present in the loaded set.
    pub fn load_weights(&mut self, reader: &mut dyn Read) -> std::io::Result<()> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let count = u32::from_le_bytes(buf4) as usize;
        self.loaded_weights.clear();
        self.accumulators.clear();
        let mut buf8 = [0u8; 8];
        for _ in 0..count {
            reader.read_exact(&mut buf8)?;
            let idx = u64::from_le_bytes(buf8);
            reader.read_exact(&mut buf8)?;
            let val = f64::from_le_bytes(buf8);
            self.loaded_weights.insert(idx, val);
        }
        Ok(())
    }

    /// Returns true if this model has loaded (frozen) weights.
    pub fn has_loaded_weights(&self) -> bool {
        !self.loaded_weights.is_empty()
    }

    /// Softmax update: treat candidates as a multi-class problem.
    /// `gold_index` is the index into `candidates` that should win.
    /// Gradient for each candidate i: softmax_prob_i - (1 if i == gold, else 0).
    pub fn update_softmax(&mut self, candidates: &[Vec<Feature>], gold_index: usize) {
        if candidates.is_empty() {
            return;
        }
        // Compute logits
        let logits: Vec<f64> = candidates.iter().map(|f| self.predict(f)).collect();
        // Numerically stable softmax
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|&e| e / sum_exp).collect();

        // Update each candidate with gradient = prob - target
        for (i, features) in candidates.iter().enumerate() {
            let target = if i == gold_index { 1.0 } else { 0.0 };
            let g = probs[i] - target;
            for f in features {
                if self.frozen.contains(&f.index) { continue; }
                let g_i = g * f.value;
                let (z_old, n_old) = self.accumulators.get(&f.index).copied().unwrap_or((0.0, 0.0));
                let w_i = if z_old.abs() <= self.l1 {
                    0.0
                } else {
                    let sign = if z_old > 0.0 { 1.0 } else { -1.0 };
                    -(z_old - sign * self.l1) / ((self.beta + n_old.sqrt()) / self.alpha + self.l2)
                };
                let sigma = ((n_old + g_i * g_i).sqrt() - n_old.sqrt()) / self.alpha;
                let z_new = z_old + g_i - sigma * w_i;
                let n_new = n_old + g_i * g_i;
                self.accumulators.insert(f.index, (z_new, n_new));
            }
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    if x < 0.0 {
        let exp = x.exp();
        exp / (1.0 + exp)
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features_from_dense(values: &[f64]) -> Vec<Feature> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| Feature {
                index: i as u64,
                value: v,
            })
            .collect()
    }

    #[test]
    fn learns_simple_pattern() {
        let mut ftrl = SparseFtrl::new(1.0, 1.0, 0.0001, 0.001);

        // Positive examples: high feature 0
        // Negative examples: low feature 0
        for _ in 0..50 {
            ftrl.update(&features_from_dense(&[1.0, 0.8, 0.3]), true);
            ftrl.update(&features_from_dense(&[1.0, 0.2, 0.7]), false);
        }

        let pos_prob = ftrl.predict_prob(&features_from_dense(&[1.0, 0.9, 0.2]));
        let neg_prob = ftrl.predict_prob(&features_from_dense(&[1.0, 0.1, 0.8]));

        assert!(
            pos_prob > neg_prob,
            "positive pattern ({pos_prob:.3}) should score higher than negative ({neg_prob:.3})"
        );
    }

    #[test]
    fn sparse_features_work() {
        let mut ftrl = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);

        // Feature 1000 = positive signal, feature 2000 = negative
        for _ in 0..30 {
            ftrl.update(
                &[
                    Feature { index: 0, value: 1.0 },
                    Feature { index: 1000, value: 1.0 },
                ],
                true,
            );
            ftrl.update(
                &[
                    Feature { index: 0, value: 1.0 },
                    Feature { index: 2000, value: 1.0 },
                ],
                false,
            );
        }

        let w1000 = ftrl.weight_at(1000);
        let w2000 = ftrl.weight_at(2000);
        assert!(w1000 > 0.0, "feature 1000 should have positive weight: {w1000}");
        assert!(w2000 < 0.0, "feature 2000 should have negative weight: {w2000}");
    }

    #[test]
    fn l1_suppresses_uninformative_features() {
        let mut ftrl = SparseFtrl::new(1.0, 1.0, 0.5, 0.001);

        // Train with only feature 1 being informative
        for _ in 0..50 {
            ftrl.update(&features_from_dense(&[1.0, 0.9, 0.5, 0.5]), true);
            ftrl.update(&features_from_dense(&[1.0, 0.1, 0.5, 0.5]), false);
        }

        let w1 = ftrl.weight_at(1).abs();
        let w2 = ftrl.weight_at(2).abs();
        let w3 = ftrl.weight_at(3).abs();
        // Informative feature should dominate uninformative ones
        assert!(
            w1 > w2 * 3.0 && w1 > w3 * 3.0,
            "feature 1 ({w1:.3}) should dominate features 2 ({w2:.3}) and 3 ({w3:.3})"
        );
    }

    #[test]
    fn weight_roundtrip_preserves_values() {
        let mut ftrl = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        for _ in 0..30 {
            ftrl.update(&features_from_dense(&[1.0, 0.8, 0.3, 0.1]), true);
            ftrl.update(&features_from_dense(&[1.0, 0.2, 0.7, 0.9]), false);
        }
        // Also add sparse features
        ftrl.update(
            &[Feature { index: 5000, value: 1.0 }, Feature { index: 9999, value: 0.5 }],
            true,
        );

        let original_weights = ftrl.weights();
        assert!(!original_weights.is_empty(), "should have non-zero weights");

        // Save
        let mut buf = Vec::new();
        ftrl.save_weights(&mut buf).unwrap();

        // Load into a fresh model
        let mut ftrl2 = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        ftrl2.load_weights(&mut &buf[..]).unwrap();

        // Check loaded weights match
        for &(idx, w) in &original_weights {
            let loaded_w = ftrl2.weight_for(idx);
            assert!(
                (w - loaded_w).abs() < 1e-12,
                "weight mismatch at index {idx}: original={w}, loaded={loaded_w}"
            );
        }
    }

    #[test]
    fn loaded_weights_produce_same_predictions() {
        let mut ftrl = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        for _ in 0..50 {
            ftrl.update(&features_from_dense(&[1.0, 0.8, 0.3]), true);
            ftrl.update(&features_from_dense(&[1.0, 0.2, 0.7]), false);
        }

        // Collect predictions from trained model
        let test_cases = vec![
            features_from_dense(&[1.0, 0.9, 0.2]),
            features_from_dense(&[1.0, 0.1, 0.8]),
            features_from_dense(&[1.0, 0.5, 0.5]),
        ];
        let original_probs: Vec<f64> = test_cases.iter().map(|f| ftrl.predict_prob(f)).collect();

        // Save and load
        let mut buf = Vec::new();
        ftrl.save_weights(&mut buf).unwrap();

        let mut ftrl2 = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        ftrl2.load_weights(&mut &buf[..]).unwrap();

        let loaded_probs: Vec<f64> = test_cases.iter().map(|f| ftrl2.predict_prob(f)).collect();

        for (i, (orig, loaded)) in original_probs.iter().zip(&loaded_probs).enumerate() {
            assert!(
                (orig - loaded).abs() < 1e-10,
                "prediction mismatch at case {i}: original={orig:.6}, loaded={loaded:.6}"
            );
        }
    }

    #[test]
    fn loaded_weights_override_accumulators() {
        let mut ftrl = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        // Train to get some accumulator values
        for _ in 0..20 {
            ftrl.update(&features_from_dense(&[1.0, 0.9]), true);
            ftrl.update(&features_from_dense(&[1.0, 0.1]), false);
        }
        let before = ftrl.predict_prob(&features_from_dense(&[1.0, 0.9]));

        // Now load different weights that should produce different predictions
        let mut loaded = HashMap::new();
        loaded.insert(0, 0.0); // zero out bias
        loaded.insert(1, 0.0); // zero out feature 1
        ftrl.loaded_weights = loaded;

        let after = ftrl.predict_prob(&features_from_dense(&[1.0, 0.9]));
        assert!(
            (after - 0.5).abs() < 0.01,
            "with zeroed loaded weights, prediction should be ~0.5, got {after:.6}"
        );
        assert!(
            (before - after).abs() > 0.1,
            "loaded weights should change predictions: before={before:.6}, after={after:.6}"
        );
    }

    #[test]
    fn load_clears_seed_accumulators() {
        // Bug: after seeding + loading, features NOT in the loaded set would
        // fall through to seed accumulators, producing wrong predictions.
        // load_weights must clear accumulators so only loaded weights are used.
        let mut model = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);

        // Seed: features 0, 1, 2 all get weights
        for _ in 0..50 {
            model.update(&features_from_dense(&[1.0, 0.5, 0.9]), true);
            model.update(&features_from_dense(&[1.0, 0.5, 0.1]), false);
        }
        let seeded_w2 = model.weight_at(2);
        assert!(seeded_w2.abs() > 0.1, "seed should give feature 2 a weight: {seeded_w2}");

        // Load weights that only cover features 0 and 1
        let mut trained = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        for _ in 0..50 {
            trained.update(&[
                Feature { index: 0, value: 1.0 },
                Feature { index: 1, value: 0.9 },
            ], true);
            trained.update(&[
                Feature { index: 0, value: 1.0 },
                Feature { index: 1, value: 0.1 },
            ], false);
        }
        let mut buf = Vec::new();
        trained.save_weights(&mut buf).unwrap();
        model.load_weights(&mut &buf[..]).unwrap();

        // Feature 2 must be zero — not the seed value
        let loaded_w2 = model.weight_for(2);
        assert!(
            loaded_w2.abs() < 1e-12,
            "feature 2 should be zero after loading (seed cleared), got {loaded_w2} (was {seeded_w2})"
        );
    }

    #[test]
    fn seed_then_load_replaces_seed_predictions() {
        // Simulate what TwoStageJudge does: seed, then load
        let mut ftrl = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);

        // "Seed" with some training
        for _ in 0..20 {
            ftrl.update(&features_from_dense(&[1.0, 0.8, 0.3]), true);
            ftrl.update(&features_from_dense(&[1.0, 0.2, 0.7]), false);
        }
        let seeded_prob = ftrl.predict_prob(&features_from_dense(&[1.0, 0.9, 0.2]));

        // Now train a separate model (simulating offline training)
        let mut trained = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        // Seed it too (same as production)
        for _ in 0..20 {
            trained.update(&features_from_dense(&[1.0, 0.8, 0.3]), true);
            trained.update(&features_from_dense(&[1.0, 0.2, 0.7]), false);
        }
        // Then train more
        for _ in 0..50 {
            trained.update(&features_from_dense(&[1.0, 0.95, 0.1]), true);
            trained.update(&features_from_dense(&[1.0, 0.05, 0.9]), false);
        }
        let trained_prob = trained.predict_prob(&features_from_dense(&[1.0, 0.9, 0.2]));

        // Save trained weights
        let mut buf = Vec::new();
        trained.save_weights(&mut buf).unwrap();

        // Load into the seeded model (load_weights clears accumulators)
        ftrl.load_weights(&mut &buf[..]).unwrap();
        let after_load_prob = ftrl.predict_prob(&features_from_dense(&[1.0, 0.9, 0.2]));

        // The loaded model uses frozen weights (from weight_for lookup),
        // NOT the FTRL formula on accumulators. So predictions won't exactly
        // match the trained model (which still computes via accumulators).
        // But they should be very close.
        assert!(
            (after_load_prob - trained_prob).abs() < 1e-6,
            "after loading, prediction should match trained model: trained={trained_prob:.6}, loaded={after_load_prob:.6}, seeded={seeded_prob:.6}"
        );
    }

    #[test]
    fn save_load_exact_equivalence_with_sparse() {
        // Full simulation: seed + train with sparse features, save, load into
        // seeded model, verify ALL predictions match (dense AND sparse inputs).
        let mut trained = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        // Seed
        for _ in 0..20 {
            trained.update(&features_from_dense(&[1.0, 0.8, 0.3]), true);
            trained.update(&features_from_dense(&[1.0, 0.2, 0.7]), false);
        }
        // Train with sparse context features (like hashed bigrams)
        for _ in 0..30 {
            trained.update(
                &[
                    Feature { index: 0, value: 1.0 },
                    Feature { index: 1, value: 0.9 },
                    Feature { index: 5000, value: 1.0 },  // sparse context
                    Feature { index: 12345, value: 1.0 },
                ],
                true,
            );
            trained.update(
                &[
                    Feature { index: 0, value: 1.0 },
                    Feature { index: 1, value: 0.1 },
                    Feature { index: 6000, value: 1.0 },
                    Feature { index: 54321, value: 1.0 },
                ],
                false,
            );
        }

        // Collect predictions on various inputs
        let test_inputs = vec![
            features_from_dense(&[1.0, 0.9, 0.2]),
            features_from_dense(&[1.0, 0.1, 0.8]),
            vec![Feature { index: 0, value: 1.0 }, Feature { index: 5000, value: 1.0 }],
            vec![Feature { index: 0, value: 1.0 }, Feature { index: 6000, value: 1.0 }],
            vec![Feature { index: 0, value: 1.0 }, Feature { index: 99999, value: 1.0 }],  // unseen sparse
        ];
        let trained_preds: Vec<f64> = test_inputs.iter().map(|f| trained.predict_prob(f)).collect();

        // Save
        let mut buf = Vec::new();
        trained.save_weights(&mut buf).unwrap();

        // Load into a fresh seeded model (simulating TwoStageJudge::new)
        let mut loaded = SparseFtrl::new(0.5, 1.0, 0.0001, 0.001);
        for _ in 0..20 {
            loaded.update(&features_from_dense(&[1.0, 0.8, 0.3]), true);
            loaded.update(&features_from_dense(&[1.0, 0.2, 0.7]), false);
        }
        loaded.load_weights(&mut &buf[..]).unwrap();

        let loaded_preds: Vec<f64> = test_inputs.iter().map(|f| loaded.predict_prob(f)).collect();

        for (i, (t, l)) in trained_preds.iter().zip(&loaded_preds).enumerate() {
            assert!(
                (t - l).abs() < 1e-6,
                "prediction mismatch at test case {i}: trained={t:.6}, loaded={l:.6}"
            );
        }
    }
}
