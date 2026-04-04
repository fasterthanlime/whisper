import type {
  AlignedWord,
  JudgeOptionDebug,
  RetrievalPrototypeTeachingCase,
  SpanDebugTrace,
} from "../beeml.generated";

export interface TeachingChoiceRow {
  id: string;
  span: SpanDebugTrace;
  option: JudgeOptionDebug;
  preview: {
    before: string;
    focus: string;
    after: string;
    full: string;
  };
  isGold: boolean;
}

function transcriptTokens(transcript: string) {
  return transcript
    .trim()
    .split(/\s+/)
    .map((word) => word.trim())
    .filter(Boolean);
}

export function normalizeComparableText(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function buildSentencePreview(
  transcript: string,
  tokenStart: number,
  tokenEnd: number,
  replacementText: string,
) {
  const tokens = transcriptTokens(transcript);
  const before = tokens.slice(0, tokenStart).join(" ");
  const focus = replacementText;
  const after = tokens.slice(tokenEnd).join(" ");
  const full = [before, focus, after].filter(Boolean).join(" ");
  return {
    before,
    focus,
    after,
    full,
  };
}

export function optionMatchesExpectedSentence(
  transcript: string,
  span: SpanDebugTrace,
  option: JudgeOptionDebug,
  teachingCase: RetrievalPrototypeTeachingCase,
) {
  const replacementText = option.is_keep_original ? span.span.text : option.term;
  const preview = buildSentencePreview(
    transcript,
    span.span.token_start,
    span.span.token_end,
    replacementText,
  );
  return (
    normalizeComparableText(preview.full) ===
    normalizeComparableText(teachingCase.source_text)
  );
}

export function spansOverlap(a: SpanDebugTrace, b: SpanDebugTrace) {
  return (
    a.span.token_start < b.span.token_end && b.span.token_start < a.span.token_end
  );
}

export function buildOverlapGroups(spans: SpanDebugTrace[]) {
  const groups: SpanDebugTrace[][] = [];

  for (const span of spans) {
    const overlappingGroups = groups.filter((group) =>
      group.some((candidate) => spansOverlap(candidate, span)),
    );

    if (overlappingGroups.length === 0) {
      groups.push([span]);
      continue;
    }

    const merged = [span];
    for (const group of overlappingGroups) {
      merged.push(...group);
    }

    for (const group of overlappingGroups) {
      const index = groups.indexOf(group);
      if (index >= 0) groups.splice(index, 1);
    }

    merged.sort(
      (a, b) =>
        a.span.token_start - b.span.token_start ||
        a.span.token_end - b.span.token_end,
    );
    groups.push(merged);
  }

  return groups;
}

export function buildTeachingChoices(
  transcript: string,
  spans: SpanDebugTrace[],
  teachingCase: RetrievalPrototypeTeachingCase,
) {
  const bestBySentence = new Map<string, TeachingChoiceRow>();

  for (const span of spans) {
    const options = dedupeJudgeOptions(span.judge_options);
    for (const option of options) {
      const preview = buildSentencePreview(
        transcript,
        span.span.token_start,
        span.span.token_end,
        option.is_keep_original ? span.span.text : option.term,
      );
      const normalized = normalizeComparableText(preview.full);
      const candidate: TeachingChoiceRow = {
        id: `${span.span.token_start}:${span.span.token_end}:${option.alias_id ?? "keep"}`,
        span,
        option,
        preview,
        isGold: optionMatchesExpectedSentence(
          transcript,
          span,
          option,
          teachingCase,
        ),
      };
      const existing = bestBySentence.get(normalized);
      if (
        !existing ||
        candidate.isGold ||
        candidate.option.probability > existing.option.probability
      ) {
        bestBySentence.set(normalized, candidate);
      }
    }
  }

  return [...bestBySentence.values()].sort((a, b) => {
    if (a.isGold !== b.isGold) return a.isGold ? -1 : 1;
    if (a.option.is_keep_original !== b.option.is_keep_original) {
      return a.option.is_keep_original ? -1 : 1;
    }
    return b.option.probability - a.option.probability || b.option.score - a.option.score;
  }).slice(0, 6);
}

export function makeApproximateWords(transcript: string): AlignedWord[] {
  const words = transcriptTokens(transcript);
  return words.map((word, index) => ({
    word,
    start: index * 0.4,
    end: index * 0.4 + 0.35,
  }));
}

export function dedupeJudgeOptions(options: JudgeOptionDebug[]) {
  const keep = options.find((option) => option.is_keep_original);
  const bestByTerm = new Map<string, JudgeOptionDebug>();

  for (const option of options) {
    if (option.is_keep_original) continue;
    const existing = bestByTerm.get(option.term);
    if (
      !existing ||
      option.probability > existing.probability ||
      (option.probability === existing.probability && option.score > existing.score)
    ) {
      bestByTerm.set(option.term, option);
    }
  }

  const deduped = [...bestByTerm.values()].sort(
    (a, b) => b.probability - a.probability || b.score - a.score,
  );
  return keep ? [keep, ...deduped] : deduped;
}

export function pickFocusSpan(
  spans: SpanDebugTrace[],
  teachingCase: RetrievalPrototypeTeachingCase,
) {
  let best: { span: SpanDebugTrace; score: number } | null = null;

  for (const span of spans) {
    const options = dedupeJudgeOptions(span.judge_options);
    const keep = options.find((option) => option.is_keep_original);
    const replacements = options.filter((option) => !option.is_keep_original);
    if (replacements.length === 0) {
      continue;
    }

    const exactMatch = options.find((option) =>
      optionMatchesExpectedSentence(
        teachingCase.transcript,
        span,
        option,
        teachingCase,
      ),
    );
    const bestReplacement = replacements[0];
    let score = bestReplacement.probability;

    if (exactMatch) {
      score = exactMatch.probability + 2.0;
    } else if (keep) {
      score = bestReplacement.probability - keep.probability;
    }

    if (!best || score > best.score) {
      best = { span, score };
    }
  }

  return best?.span ?? spans[0] ?? null;
}
