use crate::text_buffer::{TextBuffer, TokenCount};
use crate::types::RotationCutStrategy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotationTextPlan {
    pub total_tokens: TokenCount,
    pub old_context_tokens: TokenCount,
    pub stable_total_tokens: TokenCount,
    pub commit_tokens: TokenCount,
    pub next_context_tokens: TokenCount,
    pub rollback_tokens: TokenCount,
}

impl RotationTextPlan {
    pub fn drain_count(self) -> TokenCount {
        self.old_context_tokens + self.commit_tokens
    }

    pub fn commit_end(self) -> TokenCount {
        self.drain_count()
    }

    pub fn stable_end(self) -> TokenCount {
        self.stable_total_tokens
    }
}

pub fn plan_rotation(
    entries: &TextBuffer,
    old_context_tokens: usize,
    rollback_tokens: TokenCount,
    context_tokens: usize,
    _requested_tokens: TokenCount,
    rotation_cut_strategy: &RotationCutStrategy,
) -> Option<RotationTextPlan> {
    if matches!(rotation_cut_strategy, RotationCutStrategy::Uncut) {
        return None;
    }

    let total_tokens = entries.len();
    let old_context_tokens = TokenCount(old_context_tokens.min(total_tokens.0));
    let stable_total_tokens = total_tokens.saturating_sub(rollback_tokens);
    if stable_total_tokens <= old_context_tokens {
        return None;
    }

    let stable_fresh_tokens = stable_total_tokens.saturating_sub(old_context_tokens);
    let desired_context_tokens = context_tokens.min(stable_fresh_tokens.0);

    let auto_commit_target = stable_fresh_tokens.0.saturating_sub(desired_context_tokens);
    let commit_target = match rotation_cut_strategy {
        RotationCutStrategy::ManualTargetCommittedTextTokens(target) => {
            auto_commit_target.min(*target as usize)
        }
        RotationCutStrategy::Qwen3 | RotationCutStrategy::Zipa => auto_commit_target,
        RotationCutStrategy::Uncut => return None,
    };

    let fresh_entries =
        TextBuffer::from_entries(entries.entries()[old_context_tokens.0..].to_vec());
    let fresh_commit_tokens = fresh_entries.snap_to_word_boundary(TokenCount(commit_target));
    if fresh_commit_tokens.0 == 0 {
        return None;
    }

    let next_context_tokens = stable_fresh_tokens.saturating_sub(fresh_commit_tokens);
    let rollback_tokens = total_tokens.saturating_sub(stable_total_tokens);

    Some(RotationTextPlan {
        total_tokens,
        old_context_tokens,
        stable_total_tokens,
        commit_tokens: fresh_commit_tokens,
        next_context_tokens,
        rollback_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_buffer::{AsrToken, TextBuffer, TokenEntry, WordStart};
    use bee_qwen3_asr::generate::TOP_K;

    fn token(id: u32, word: bool) -> TokenEntry {
        TokenEntry {
            token: AsrToken {
                id,
                concentration: 0.0,
                margin: 0.0,
                alternative_count: TOP_K as u8,
                top_ids: [0; TOP_K],
                top_logits: [0.0; TOP_K],
            },
            word: word.then_some(WordStart { alignment: None }),
        }
    }

    #[test]
    fn auto_rotation_commits_stable_prefix_and_keeps_context() {
        let entries = TextBuffer::from_entries(vec![
            token(1, true),
            token(2, false),
            token(3, true),
            token(4, false),
            token(5, true),
            token(6, false),
        ]);
        let plan = plan_rotation(
            &entries,
            0,
            TokenCount(2),
            2,
            TokenCount(4),
            &RotationCutStrategy::Qwen3,
        )
        .unwrap();

        assert_eq!(plan.commit_tokens, TokenCount(2));
        assert_eq!(plan.next_context_tokens, TokenCount(2));
        assert_eq!(plan.rollback_tokens, TokenCount(2));
    }

    #[test]
    fn manual_target_caps_commit_amount() {
        let entries = TextBuffer::from_entries(vec![
            token(1, true),
            token(2, false),
            token(3, true),
            token(4, false),
            token(5, true),
            token(6, false),
            token(7, true),
        ]);
        let plan = plan_rotation(
            &entries,
            0,
            TokenCount(1),
            0,
            TokenCount(99),
            &RotationCutStrategy::ManualTargetCommittedTextTokens(2),
        )
        .unwrap();

        assert_eq!(plan.commit_tokens, TokenCount(2));
        assert_eq!(plan.next_context_tokens, TokenCount(4));
    }
}
