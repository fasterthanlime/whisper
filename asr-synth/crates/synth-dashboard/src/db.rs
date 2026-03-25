use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS vocab (
    id INTEGER PRIMARY KEY,
    term TEXT UNIQUE NOT NULL COLLATE NOCASE,
    spoken_auto TEXT NOT NULL,
    spoken_override TEXT,
    reviewed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentences (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    spoken TEXT NOT NULL,
    vocab_terms TEXT NOT NULL,
    unknown_words TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    wav_path TEXT,
    parakeet_output TEXT,
    qwen_output TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcriptions (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    app TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',
    corrected_text TEXT,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    config TEXT,
    log TEXT NOT NULL DEFAULT '',
    result TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS candidate_sentences (
    id INTEGER PRIMARY KEY,
    text TEXT UNIQUE NOT NULL,
    spoken TEXT NOT NULL,
    vocab_terms TEXT NOT NULL,
    source TEXT NOT NULL,
    unknown_words TEXT NOT NULL DEFAULT '[]',
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_pairs (
    id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    original TEXT NOT NULL,
    qwen TEXT NOT NULL,
    parakeet TEXT NOT NULL,
    sentence TEXT NOT NULL,
    spoken TEXT NOT NULL,
    orig_alignment TEXT,
    qwen_alignment TEXT,
    parakeet_alignment TEXT,
    cons_time TEXT,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_transcriptions_dedup
    ON transcriptions(timestamp, text);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sentences_text
    ON sentences(text);
"#;

// --- Vocab ---

#[derive(Debug, Serialize, Deserialize)]
pub struct VocabRow {
    pub id: i64,
    pub term: String,
    pub spoken_auto: String,
    pub spoken_override: Option<String>,
    pub reviewed: bool,
}

impl VocabRow {
    /// The effective spoken form: override if set, otherwise auto
    pub fn spoken(&self) -> &str {
        self.spoken_override.as_deref().unwrap_or(&self.spoken_auto)
    }
}

// --- Sentences ---

#[derive(Debug, Serialize, Deserialize)]
pub struct SentenceRow {
    pub id: i64,
    pub text: String,
    pub spoken: String,
    pub vocab_terms: String, // JSON array
    pub unknown_words: String, // JSON array
    pub status: String,
    pub wav_path: Option<String>,
    pub alignment_json: Option<String>,
    pub tts_backend: Option<String>,
    pub parakeet_output: Option<String>,
    pub qwen_output: Option<String>,
    pub human_wav_path: Option<String>,
}

// --- Stats ---

#[derive(Debug, Serialize)]
pub struct CorpusStats {
    pub vocab_total: i64,
    pub vocab_reviewed: i64,
    pub vocab_with_override: i64,
    pub vocab_unreviewed: i64,
    pub candidates_total: i64,
    pub sentences_total: i64,
    pub sentences_pending: i64,
    pub sentences_approved: i64,
    pub sentences_rejected: i64,
    pub sentences_with_audio: i64,
    pub sentences_with_asr: i64,
    pub transcriptions_total: i64,
}

// --- Jobs ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: i64,
    pub job_type: String,
    pub status: String,
    pub config: Option<String>,
    pub log: String,
    pub result: Option<String>,
    pub created_at: String,
    pub finished_at: Option<String>,
}

// --- Log entry for Hark import ---

#[derive(Debug, Deserialize)]
pub struct LogEntry {
    pub text: String,
    pub timestamp: String,
    pub app: Option<String>,
}

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).context("opening database")?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .context("setting pragmas")?;
        conn.execute_batch(SCHEMA).context("creating schema")?;
        // Migrations: add columns if missing (idempotent — errors ignored for already-existing columns)
        let _ = conn.execute_batch("ALTER TABLE sentences ADD COLUMN unknown_words TEXT NOT NULL DEFAULT '[]';");
        let _ = conn.execute_batch("ALTER TABLE sentences ADD COLUMN alignment_json TEXT;");
        let _ = conn.execute_batch("ALTER TABLE sentences ADD COLUMN tts_backend TEXT;");
        let _ = conn.execute_batch("ALTER TABLE sentences ADD COLUMN human_wav_path TEXT;");
        let _ = conn.execute_batch("ALTER TABLE vocab ADD COLUMN curated TEXT;"); // 'kept', 'removed', or NULL
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN orig_alignment TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN qwen_alignment TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN parakeet_alignment TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN cons_time TEXT;");

        // Migration: case-insensitive dedup of vocab terms.
        // For each group of case-insensitive duplicates, keep the row that has
        // the most info (reviewed > unreviewed, has override > no override, lowest id as tiebreaker).
        // Then rebuild the table with COLLATE NOCASE on the unique constraint.
        Self::migrate_vocab_nocase(&conn);

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vocab_confusions (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                qwen_heard TEXT NOT NULL,
                parakeet_heard TEXT NOT NULL,
                qwen_match INTEGER NOT NULL,
                parakeet_match INTEGER NOT NULL,
                tts_backend TEXT,
                created_at TEXT NOT NULL
            )"
        ).ok();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS import_offsets (
                source TEXT PRIMARY KEY,
                byte_offset INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )"
        ).ok();
        Ok(Db { conn })
    }

    // ==================== IMPORT OFFSETS ====================

    pub fn get_import_offset(&self, source: &str) -> Result<u64> {
        let result = self.conn.query_row(
            "SELECT byte_offset FROM import_offsets WHERE source = ?1",
            params![source],
            |r| r.get::<_, i64>(0),
        );
        match result {
            Ok(v) => Ok(v as u64),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(0),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_import_offset(&self, source: &str, offset: u64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO import_offsets (source, byte_offset, updated_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(source) DO UPDATE SET byte_offset = excluded.byte_offset, updated_at = excluded.updated_at",
            params![source, offset as i64, now_str()],
        )?;
        Ok(())
    }

    // ==================== VOCAB ====================

    /// Seed the vocab table from the hardcoded PRONUNCIATION_OVERRIDES.
    /// Only inserts terms that don't already exist.
    pub fn seed_overrides(&self) -> Result<usize> {
        let now = now_str();
        let mut count = 0;
        for &(term, spoken) in synth_textgen::corpus::PRONUNCIATION_OVERRIDES {
            let auto = synth_textgen::corpus::to_spoken(term);
            let result = self.conn.execute(
                "INSERT OR IGNORE INTO vocab (term, spoken_auto, spoken_override, reviewed, created_at, updated_at)
                 VALUES (?1, ?2, ?3, 0, ?4, ?4)",
                params![term, auto, spoken, now],
            );
            if let Ok(n) = result {
                count += n;
            }
        }
        Ok(count)
    }

    /// Import extracted vocab entries into the table. Sets spoken_auto but
    /// does NOT overwrite existing spoken_override values.
    pub fn import_vocab(&self, entries: &[synth_textgen::corpus::VocabEntry]) -> Result<usize> {
        let now = now_str();
        let mut count = 0;
        for entry in entries {
            let result = self.conn.execute(
                "INSERT INTO vocab (term, spoken_auto, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?3)
                 ON CONFLICT(term) DO UPDATE SET spoken_auto = excluded.spoken_auto, updated_at = excluded.updated_at",
                params![entry.term, entry.spoken, now],
            );
            if let Ok(n) = result {
                count += n;
            }
        }
        Ok(count)
    }

    pub fn list_vocab(
        &self,
        search: Option<&str>,
        reviewed_only: Option<bool>,
        has_override: Option<bool>,
        sort_recent: bool,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<VocabRow>> {
        let mut conditions = Vec::new();
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        // Exclude tombstoned (removed) terms by default
        conditions.push("(curated IS NULL OR curated != 'removed')".to_string());

        if let Some(q) = search {
            conditions.push(format!("term LIKE ?{idx}"));
            param_values.push(Box::new(format!("%{q}%")));
            idx += 1;
        }
        if let Some(rev) = reviewed_only {
            conditions.push(format!("reviewed = ?{idx}"));
            param_values.push(Box::new(rev as i64));
            idx += 1;
        }
        if let Some(true) = has_override {
            conditions.push("spoken_override IS NOT NULL AND spoken_override != ''".to_string());
        } else if let Some(false) = has_override {
            conditions.push("(spoken_override IS NULL OR spoken_override = '')".to_string());
        }

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let sql = format!(
            "SELECT id, term, spoken_auto, spoken_override, reviewed FROM vocab {where_clause} ORDER BY {} LIMIT ?{idx} OFFSET ?{}",
            if sort_recent { "updated_at DESC" } else { "term COLLATE NOCASE" }, idx + 1
        );
        param_values.push(Box::new(limit));
        param_values.push(Box::new(offset));

        let refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(refs.as_slice(), |row| {
            Ok(VocabRow {
                id: row.get(0)?,
                term: row.get(1)?,
                spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?,
                reviewed: row.get::<_, i64>(4)? != 0,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Insert a vocab term discovered by G2P failure. No override, not reviewed.
    pub fn insert_candidate_vocab(&self, term: &str, spoken_auto: &str) -> Result<bool> {
        let now = now_str();
        let result = self.conn.execute(
            "INSERT OR IGNORE INTO vocab (term, spoken_auto, reviewed, created_at, updated_at) VALUES (?1, ?2, 0, ?3, ?3)",
            params![term, spoken_auto, now],
        )?;
        Ok(result > 0)
    }

    pub fn vocab_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM vocab", [], |r| r.get(0))?)
    }

    pub fn update_vocab_override(&self, id: i64, spoken_override: Option<&str>) -> Result<()> {
        self.conn.execute(
            "UPDATE vocab SET spoken_override = ?1, updated_at = ?2 WHERE id = ?3",
            params![spoken_override, now_str(), id],
        )?;
        Ok(())
    }

    pub fn set_vocab_reviewed(&self, id: i64, reviewed: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE vocab SET reviewed = ?1, updated_at = ?2 WHERE id = ?3",
            params![reviewed as i64, now_str(), id],
        )?;
        Ok(())
    }

    /// Get a map of term → effective spoken form for all vocab with overrides
    pub fn get_spoken_overrides(&self) -> Result<HashMap<String, String>> {
        let mut stmt = self.conn.prepare(
            "SELECT term, COALESCE(spoken_override, spoken_auto) FROM vocab WHERE spoken_override IS NOT NULL",
        )?;
        let mut map = HashMap::new();
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (term, spoken) = row?;
            map.insert(term, spoken);
        }
        Ok(map)
    }

    // ==================== SENTENCES ====================

    pub fn insert_sentences(&self, sentences: &[synth_textgen::templates::GeneratedSentence]) -> Result<usize> {
        let now = now_str();
        let mut count = 0;
        for s in sentences {
            let vocab_json = serde_json::to_string(&s.vocab_terms)?;
            let unknown_json = serde_json::to_string(&crate::tts::detect_unknown_words(&s.text))?;
            self.conn.execute(
                "INSERT INTO sentences (text, spoken, vocab_terms, unknown_words, status, created_at) VALUES (?1, ?2, ?3, ?4, 'pending', ?5)",
                params![s.text, s.spoken, vocab_json, unknown_json, now],
            )?;
            count += 1;
        }
        Ok(count)
    }

    /// Insert a sentence, skipping duplicates.
    pub fn insert_sentence_from_candidate(&self, text: &str, spoken: &str, vocab_terms: &str, unknown_words: &str) -> Result<bool> {
        let now = now_str();
        let n = self.conn.execute(
            "INSERT OR IGNORE INTO sentences (text, spoken, vocab_terms, unknown_words, status, created_at) VALUES (?1, ?2, ?3, ?4, 'pending', ?5)",
            params![text, spoken, vocab_terms, unknown_words, now],
        )?;
        Ok(n > 0)
    }

    pub fn list_sentences(
        &self,
        status: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SentenceRow>> {
        let cols = "id, text, spoken, vocab_terms, unknown_words, status, wav_path, alignment_json, tts_backend, parakeet_output, qwen_output, human_wav_path";
        let (sql, param_values): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match status {
            Some(s) => (
                format!("SELECT {cols} FROM sentences WHERE status = ?1 ORDER BY id LIMIT ?2 OFFSET ?3"),
                vec![Box::new(s.to_string()), Box::new(limit), Box::new(offset)],
            ),
            None => (
                format!("SELECT {cols} FROM sentences ORDER BY id LIMIT ?1 OFFSET ?2"),
                vec![Box::new(limit), Box::new(offset)],
            ),
        };
        let refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(refs.as_slice(), |row| {
            Ok(SentenceRow {
                id: row.get(0)?,
                text: row.get(1)?,
                spoken: row.get(2)?,
                vocab_terms: row.get(3)?,
                unknown_words: row.get(4)?,
                status: row.get(5)?,
                wav_path: row.get(6)?,
                alignment_json: row.get(7)?,
                tts_backend: row.get(8)?,
                parakeet_output: row.get(9)?,
                qwen_output: row.get(10)?,
                human_wav_path: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_sentence_status(&self, id: i64, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET status = ?1 WHERE id = ?2",
            params![status, id],
        )?;
        Ok(())
    }

    pub fn update_sentence_spoken(&self, id: i64, spoken: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET spoken = ?1, status = 'needs_resynth' WHERE id = ?2",
            params![spoken, id],
        )?;
        Ok(())
    }

    /// Update sentence text, spoken form, and unknown words (for correcting transcription errors).
    pub fn update_sentence_text(&self, id: i64, text: &str, spoken: &str, unknown_words: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET text = ?1, spoken = ?2, unknown_words = ?3, status = 'pending' WHERE id = ?4",
            params![text, spoken, unknown_words, id],
        )?;
        Ok(())
    }

    pub fn update_sentence_wav(&self, id: i64, wav_path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET wav_path = ?1 WHERE id = ?2",
            params![wav_path, id],
        )?;
        Ok(())
    }

    pub fn update_sentence_asr(&self, id: i64, parakeet: &str, qwen: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET parakeet_output = ?1, qwen_output = ?2 WHERE id = ?3",
            params![parakeet, qwen, id],
        )?;
        Ok(())
    }

    /// Get a single sentence by ID.
    pub fn get_sentence(&self, id: i64) -> Result<Option<SentenceRow>> {
        let cols = "id, text, spoken, vocab_terms, unknown_words, status, wav_path, alignment_json, tts_backend, parakeet_output, qwen_output, human_wav_path";
        let mut stmt = self.conn.prepare(&format!("SELECT {cols} FROM sentences WHERE id = ?1"))?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(SentenceRow {
                id: row.get(0)?, text: row.get(1)?, spoken: row.get(2)?,
                vocab_terms: row.get(3)?, unknown_words: row.get(4)?, status: row.get(5)?,
                wav_path: row.get(6)?, alignment_json: row.get(7)?, tts_backend: row.get(8)?,
                parakeet_output: row.get(9)?, qwen_output: row.get(10)?, human_wav_path: row.get(11)?,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Get IDs of pending sentences, prioritizing those with unknown words.
    pub fn pending_sentence_ids(&self, limit: i64) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM sentences WHERE status = 'pending' \
             ORDER BY (CASE WHEN unknown_words != '[]' THEN 0 ELSE 1 END), RANDOM() LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Find a vocab entry by term (exact, case-insensitive).
    pub fn delete_vocab_by_term(&self, term: &str) -> Result<()> {
        self.conn.execute("DELETE FROM vocab WHERE LOWER(term) = LOWER(?1)", params![term])?;
        Ok(())
    }

    pub fn find_vocab_by_term(&self, term: &str) -> Result<Option<VocabRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, spoken_auto, spoken_override, reviewed FROM vocab WHERE LOWER(term) = LOWER(?1)"
        )?;
        let mut rows = stmt.query_map(params![term], |row| {
            Ok(VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// Auto-resolve an unknown word: set spoken_override = lowercase term if no override exists.
    pub fn auto_resolve_unknown(&self, term: &str) -> Result<()> {
        let now = now_str();
        self.conn.execute(
            "UPDATE vocab SET spoken_override = LOWER(term), updated_at = ?1 WHERE LOWER(term) = LOWER(?2) AND spoken_override IS NULL",
            params![now, term],
        )?;
        Ok(())
    }

    /// Update precomputed TTS + alignment data for a sentence.
    pub fn update_sentence_precomputed(
        &self, id: i64, wav_path: &str, alignment_json: &str, backend: &str, spoken: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET wav_path = ?1, alignment_json = ?2, tts_backend = ?3, spoken = ?4 WHERE id = ?5",
            params![wav_path, alignment_json, backend, spoken, id],
        )?;
        Ok(())
    }

    /// Count sentences by status.
    pub fn sentence_count_by_status(&self) -> Result<(i64, i64, i64)> {
        let approved: i64 = self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE status = 'approved'", [], |r| r.get(0))?;
        let rejected: i64 = self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE status = 'rejected'", [], |r| r.get(0))?;
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM sentences", [], |r| r.get(0))?;
        Ok((approved, rejected, total))
    }

    /// All sentence texts (for Markov chain building).
    // ==================== CORPUS PAIRS ====================

    pub fn insert_corpus_pair(
        &self, term: &str, original: &str, qwen: &str, parakeet: &str,
        sentence: &str, spoken: &str,
        orig_align: Option<&str>, qwen_align: Option<&str>, parakeet_align: Option<&str>,
        cons_time: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO corpus_pairs (term, original, qwen, parakeet, sentence, spoken, orig_alignment, qwen_alignment, parakeet_alignment, cons_time, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![term, original, qwen, parakeet, sentence, spoken, orig_align, qwen_align, parakeet_align, cons_time, now_str()],
        )?;
        Ok(())
    }

    /// Count existing passes per term in corpus_pairs.
    pub fn corpus_passes_per_term(&self) -> Result<HashMap<String, usize>> {
        let mut stmt = self.conn.prepare("SELECT term, COUNT(*) FROM corpus_pairs GROUP BY term")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })?;
        let mut map = HashMap::new();
        for row in rows {
            let (term, count) = row?;
            map.insert(term, count);
        }
        Ok(map)
    }

    pub fn corpus_pair_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM corpus_pairs", [], |r| r.get(0))?)
    }

    pub fn corpus_pairs_all(&self) -> Result<Vec<serde_json::Value>> {
        let mut stmt = self.conn.prepare(
            "SELECT term, original, qwen, parakeet, sentence, spoken, orig_alignment, qwen_alignment, parakeet_alignment, cons_time FROM corpus_pairs ORDER BY term COLLATE NOCASE, id"
        )?;
        let rows = stmt.query_map([], |row| {
            let orig_align: Option<String> = row.get(6)?;
            let qwen_align: Option<String> = row.get(7)?;
            let para_align: Option<String> = row.get(8)?;
            let cons_time: Option<String> = row.get(9)?;
            Ok(serde_json::json!({
                "term": row.get::<_, String>(0)?,
                "original": row.get::<_, String>(1)?,
                "qwen": row.get::<_, String>(2)?,
                "parakeet": row.get::<_, String>(3)?,
                "sentence": row.get::<_, String>(4)?,
                "spoken": row.get::<_, String>(5)?,
                "clean": true,
                "orig_alignment": orig_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "qwen_alignment": qwen_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "parakeet_alignment": para_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "cons_time": cons_time.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
            }))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn reset_corpus(&self) -> Result<()> {
        self.conn.execute("DELETE FROM corpus_pairs", [])?;
        Ok(())
    }

    /// Get unique clean triplets for evaluation (deduplicated by original+qwen+parakeet).
    pub fn corpus_unique_triplets(&self) -> Result<Vec<(String, String, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT term, original, qwen, parakeet FROM corpus_pairs"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn all_sentence_texts(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT text FROM sentences UNION SELECT text FROM candidate_sentences")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn list_approved_sentences(&self) -> Result<Vec<SentenceRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, spoken, vocab_terms, unknown_words, status, wav_path, alignment_json, tts_backend, parakeet_output, qwen_output, human_wav_path FROM sentences WHERE status = 'approved' ORDER BY id"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(SentenceRow {
                id: row.get(0)?,
                text: row.get(1)?,
                spoken: row.get(2)?,
                vocab_terms: row.get(3)?,
                unknown_words: row.get(4)?,
                status: row.get(5)?,
                wav_path: row.get(6)?,
                alignment_json: row.get(7)?,
                tts_backend: row.get(8)?,
                parakeet_output: row.get(9)?,
                qwen_output: row.get(10)?,
                human_wav_path: row.get(11)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn sentences_with_human_recording_count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM sentences WHERE human_wav_path IS NOT NULL",
            [], |r| r.get(0),
        )?)
    }

    pub fn update_sentence_human_wav(&self, id: i64, path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE sentences SET human_wav_path = ?1 WHERE id = ?2",
            params![path, id],
        )?;
        Ok(())
    }

    // ==================== CONFUSIONS ====================

    pub fn insert_confusion(&self, term: &str, qwen: &str, parakeet: &str, qwen_match: bool, parakeet_match: bool, backend: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO vocab_confusions (term, qwen_heard, parakeet_heard, qwen_match, parakeet_match, tts_backend, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![term, qwen, parakeet, qwen_match as i64, parakeet_match as i64, backend, now_str()],
        )?;
        Ok(())
    }

    pub fn clear_confusions(&self) -> Result<()> {
        self.conn.execute("DELETE FROM vocab_confusions", [])?;
        Ok(())
    }

    /// Get vocab terms sorted by how often ASR gets them wrong
    pub fn vocab_scan_results(&self) -> Result<Vec<(String, i64, i64, i64)>> {
        // Returns (term, total_scans, qwen_errors, parakeet_errors)
        let mut stmt = self.conn.prepare(
            "SELECT term, COUNT(*) as total,
                    SUM(CASE WHEN qwen_match = 0 THEN 1 ELSE 0 END) as qwen_err,
                    SUM(CASE WHEN parakeet_match = 0 THEN 1 ELSE 0 END) as parakeet_err
             FROM vocab_confusions
             GROUP BY term
             ORDER BY (qwen_err + parakeet_err) DESC, term"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get all unique mishearings for a term
    pub fn confusions_for_term(&self, term: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT qwen_heard, parakeet_heard FROM vocab_confusions WHERE term = ?1 AND (qwen_match = 0 OR parakeet_match = 0)"
        )?;
        let rows = stmt.query_map(params![term], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Terms that haven't been curated yet (curated IS NULL)
    pub fn uncurated_vocab_terms(&self) -> Result<Vec<(String, Option<String>)>> {
        let mut stmt = self.conn.prepare("SELECT term, spoken_override FROM vocab WHERE curated IS NULL ORDER BY term COLLATE NOCASE")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get IDs of kept vocab terms that haven't been reviewed yet
    pub fn unreviewed_vocab_ids(&self, limit: i64) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT id FROM vocab WHERE curated = 'kept' AND reviewed = 0 ORDER BY id LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_vocab(&self, id: i64) -> Result<Option<VocabRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, spoken_auto, spoken_override, reviewed FROM vocab WHERE id = ?1"
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// All reviewed (approved) vocab terms for corpus generation.
    pub fn list_reviewed_vocab(&self) -> Result<Vec<VocabRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, spoken_auto, spoken_override, reviewed FROM vocab WHERE reviewed = 1 AND (curated IS NULL OR curated = 'kept')"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn mark_vocab_reviewed(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE vocab SET reviewed = 1, updated_at = ?1 WHERE id = ?2",
            params![now_str(), id],
        )?;
        Ok(())
    }

    pub fn reject_vocab(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE vocab SET curated = 'removed', updated_at = ?1 WHERE id = ?2",
            params![now_str(), id],
        )?;
        Ok(())
    }

    pub fn vocab_review_counts(&self) -> Result<(i64, i64, i64)> {
        // (reviewed, unreviewed, total kept)
        let reviewed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vocab WHERE curated = 'kept' AND reviewed = 1", [], |r| r.get(0))?;
        let unreviewed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vocab WHERE curated = 'kept' AND reviewed = 0", [], |r| r.get(0))?;
        Ok((reviewed, unreviewed, reviewed + unreviewed))
    }

    pub fn set_vocab_curated(&self, term: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE vocab SET curated = ?1, updated_at = ?2 WHERE LOWER(term) = LOWER(?3)",
            params![status, now_str(), term],
        )?;
        Ok(())
    }

    pub fn all_vocab_terms(&self) -> Result<Vec<(String, Option<String>)>> {
        // Returns (term, spoken_override) — only kept or uncurated terms (excludes removed)
        let mut stmt = self.conn.prepare("SELECT term, spoken_override FROM vocab WHERE curated IS NULL OR curated = 'kept' ORDER BY term COLLATE NOCASE")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn confusion_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(DISTINCT term) FROM vocab_confusions", [], |r| r.get(0))?)
    }

    // ==================== CANDIDATES ====================

    pub fn insert_candidate(&self, text: &str, spoken: &str, vocab_terms: &str, source: &str, unknown_words: &str) -> Result<bool> {
        let result = self.conn.execute(
            "INSERT OR IGNORE INTO candidate_sentences (text, spoken, vocab_terms, source, unknown_words, imported_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![text, spoken, vocab_terms, source, unknown_words, now_str()],
        )?;
        Ok(result > 0)
    }

    pub fn candidate_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM candidate_sentences", [], |r| r.get(0))?)
    }

    /// Pick N candidates that aren't already in the sentences table.
    /// If `prioritize_unknown` is true, prefer sentences with unknown words.
    pub fn pick_candidates(&self, count: i64, prioritize_unknown: bool) -> Result<Vec<(String, String, String, String)>> {
        let sql = if prioritize_unknown {
            // Sentences with unknown words first, then random
            "SELECT c.text, c.spoken, c.vocab_terms, c.unknown_words FROM candidate_sentences c
             WHERE c.text NOT IN (SELECT text FROM sentences)
             ORDER BY (CASE WHEN c.unknown_words != '[]' THEN 0 ELSE 1 END), RANDOM()
             LIMIT ?1"
        } else {
            "SELECT c.text, c.spoken, c.vocab_terms, c.unknown_words FROM candidate_sentences c
             WHERE c.text NOT IN (SELECT text FROM sentences)
             ORDER BY RANDOM() LIMIT ?1"
        };
        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![count], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ==================== STATS ====================

    pub fn stats(&self) -> Result<CorpusStats> {
        Ok(CorpusStats {
            vocab_total: self.conn.query_row("SELECT COUNT(*) FROM vocab", [], |r| r.get(0))?,
            vocab_reviewed: self.conn.query_row("SELECT COUNT(*) FROM vocab WHERE reviewed = 1", [], |r| r.get(0))?,
            vocab_with_override: self.conn.query_row("SELECT COUNT(*) FROM vocab WHERE spoken_override IS NOT NULL", [], |r| r.get(0))?,
            vocab_unreviewed: self.conn.query_row("SELECT COUNT(*) FROM vocab WHERE curated = 'kept' AND reviewed = 0", [], |r| r.get(0))?,
            candidates_total: self.conn.query_row("SELECT COUNT(*) FROM candidate_sentences", [], |r| r.get(0))?,
            sentences_total: self.conn.query_row("SELECT COUNT(*) FROM sentences", [], |r| r.get(0))?,
            sentences_pending: self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE status = 'pending'", [], |r| r.get(0))?,
            sentences_approved: self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE status = 'approved'", [], |r| r.get(0))?,
            sentences_rejected: self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE status = 'rejected'", [], |r| r.get(0))?,
            sentences_with_audio: self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE wav_path IS NOT NULL", [], |r| r.get(0))?,
            sentences_with_asr: self.conn.query_row("SELECT COUNT(*) FROM sentences WHERE parakeet_output IS NOT NULL", [], |r| r.get(0))?,
            transcriptions_total: self.conn.query_row("SELECT COUNT(*) FROM transcriptions", [], |r| r.get(0))?,
        })
    }

    // ==================== HARK IMPORT ====================

    pub fn import_hark_log(&self, path: &Path) -> Result<usize> {
        let content = std::fs::read_to_string(path).context("reading JSONL file")?;
        let now = now_str();
        let mut count = 0usize;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let entry: LogEntry = match serde_json::from_str(line) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let result = self.conn.execute(
                "INSERT OR IGNORE INTO transcriptions (text, timestamp, app, imported_at) VALUES (?1, ?2, ?3, ?4)",
                params![entry.text, entry.timestamp, entry.app, now],
            );
            if let Ok(changed) = result { count += changed; }
        }
        Ok(count)
    }

    // ==================== JOBS ====================

    pub fn create_job(&self, job_type: &str, config: Option<&str>) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO jobs (job_type, status, config, created_at) VALUES (?1, 'running', ?2, ?3)",
            params![job_type, config, now_str()],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn append_job_log(&self, job_id: i64, line: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE jobs SET log = log || ?1 || char(10) WHERE id = ?2",
            params![line, job_id],
        )?;
        Ok(())
    }

    /// Update the result field of a running job (for live stats).
    pub fn update_job_result(&self, job_id: i64, result: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE jobs SET result = ?1 WHERE id = ?2",
            params![result, job_id],
        )?;
        Ok(())
    }

    /// Mark any jobs stuck in "running" as failed (orphaned from a previous process).
    pub fn fail_orphaned_jobs(&self) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ?1 WHERE status = 'running'",
            params![now_str()],
        )?;
        if n > 0 {
            eprintln!("[db] Cleaned up {n} orphaned running job(s)");
        }
        Ok(())
    }

    pub fn finish_job(&self, job_id: i64, status: &str, result: Option<&str>) -> Result<()> {
        self.conn.execute(
            "UPDATE jobs SET status = ?1, result = ?2, finished_at = ?3 WHERE id = ?4",
            params![status, result, now_str(), job_id],
        )?;
        Ok(())
    }

    pub fn get_job(&self, job_id: i64) -> Result<Option<Job>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, job_type, status, config, log, result, created_at, finished_at FROM jobs WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![job_id], |row| {
            Ok(Job {
                id: row.get(0)?, job_type: row.get(1)?, status: row.get(2)?,
                config: row.get(3)?, log: row.get(4)?, result: row.get(5)?,
                created_at: row.get(6)?, finished_at: row.get(7)?,
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    pub fn list_jobs(&self) -> Result<Vec<Job>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, job_type, status, config, log, result, created_at, finished_at FROM jobs ORDER BY id DESC LIMIT 50",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Job {
                id: row.get(0)?, job_type: row.get(1)?, status: row.get(2)?,
                config: row.get(3)?, log: row.get(4)?, result: row.get(5)?,
                created_at: row.get(6)?, finished_at: row.get(7)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }
}

impl Db {
    /// One-time migration: deduplicate vocab terms case-insensitively and
    /// rebuild the table with COLLATE NOCASE on the UNIQUE constraint.
    fn migrate_vocab_nocase(conn: &Connection) {
        // Check if the table already has COLLATE NOCASE by inspecting the schema
        let schema: String = conn.query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vocab'",
            [], |r| r.get(0),
        ).unwrap_or_default();

        if schema.to_uppercase().contains("COLLATE NOCASE") {
            return; // Already migrated
        }

        eprintln!("[db-migrate] Rebuilding vocab table with case-insensitive uniqueness...");

        let result = conn.execute_batch("
            -- Dedup: keep the best row per case-insensitive group
            -- (reviewed > unreviewed, has override > no override, lowest id as tiebreaker)
            CREATE TABLE vocab_dedup AS
            SELECT * FROM vocab WHERE id IN (
                SELECT id FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY LOWER(term)
                               ORDER BY reviewed DESC,
                                        CASE WHEN spoken_override IS NOT NULL AND spoken_override != '' THEN 0 ELSE 1 END,
                                        id ASC
                           ) AS rn
                    FROM vocab
                )
                WHERE rn = 1
            );

            DROP TABLE vocab;

            CREATE TABLE vocab (
                id INTEGER PRIMARY KEY,
                term TEXT UNIQUE NOT NULL COLLATE NOCASE,
                spoken_auto TEXT NOT NULL,
                spoken_override TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                curated TEXT
            );

            INSERT INTO vocab (id, term, spoken_auto, spoken_override, reviewed, created_at, updated_at, curated)
            SELECT id, term, spoken_auto, spoken_override, reviewed, created_at, updated_at, curated
            FROM vocab_dedup;

            DROP TABLE vocab_dedup;
        ");
        match result {
            Ok(()) => {
                let count: i64 = conn.query_row("SELECT COUNT(*) FROM vocab", [], |r| r.get(0)).unwrap_or(0);
                eprintln!("[db-migrate] Vocab table rebuilt with COLLATE NOCASE ({count} terms)");
            }
            Err(e) => eprintln!("[db-migrate] Vocab rebuild failed (non-fatal): {e}"),
        }
    }
}

pub fn now_str() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap();
    format!("{}Z", dur.as_secs())
}
