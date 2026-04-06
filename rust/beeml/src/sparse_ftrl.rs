//! Sparse FTRL-Proximal optimizer for online linear classification.
//!
//! Only stores weights for features that have received non-zero updates,
//! making it efficient for high-dimensional hashed feature spaces.
//!
//! Reference: McMahan et al., "Ad Click Prediction: a View from the Trenches" (2013)

use std::collections::HashMap;

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
}

/// A single active feature: index + value.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub index: u64,
    pub value: f64,
}

impl SparseFtrl {
    pub fn new(alpha: f64, beta: f64, l1: f64, l2: f64) -> Self {
        Self {
            accumulators: HashMap::new(),
            alpha,
            beta,
            l1,
            l2,
            frozen: std::collections::HashSet::new(),
        }
    }

    /// Freeze a set of feature indices — they will not be updated during training.
    pub fn freeze(&mut self, indices: impl IntoIterator<Item = u64>) {
        self.frozen.extend(indices);
    }

    /// Compute the weight for a single feature from its accumulators.
    fn weight_for(&self, index: u64) -> f64 {
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
}
