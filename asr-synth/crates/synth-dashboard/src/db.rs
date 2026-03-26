use anyhow::{Context, Result};
use base64::Engine as _;
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

CREATE TABLE IF NOT EXISTS authored_sentence_recordings (
    id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    sentence TEXT NOT NULL COLLATE NOCASE,
    take_no INTEGER NOT NULL,
    wav_path TEXT NOT NULL,
    created_at TEXT NOT NULL
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
    pub description: Option<String>,
}

impl VocabRow {
    /// The effective spoken form: override if set, otherwise auto
    pub fn spoken(&self) -> &str {
        self.spoken_override.as_deref().unwrap_or(&self.spoken_auto)
    }
}

// --- Sentences ---

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalItem {
    pub term: String,
    pub original: String,
    pub qwen: String,
    pub parakeet: String,
    pub is_mistake: bool,
    pub hit_count: i64,
}

#[derive(serde::Serialize)]
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

#[derive(Debug, Serialize)]
pub struct AuthoredSentenceRecordingRow {
    pub id: i64,
    pub term: String,
    pub sentence: String,
    pub take_no: i64,
    pub wav_path: String,
    pub created_at: String,
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
        let _ = conn.execute_batch("ALTER TABLE vocab ADD COLUMN description TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN orig_alignment TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN qwen_alignment TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN parakeet_alignment TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN cons_time TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN trim_info TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN qwen_full TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN audio_ogg BLOB;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 1;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN is_mistake INTEGER NOT NULL DEFAULT 1;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN updated_at TEXT;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN rejected INTEGER NOT NULL DEFAULT 0;");
        let _ = conn.execute_batch("ALTER TABLE corpus_pairs ADD COLUMN review_status TEXT;");

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
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS authored_sentences (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                sentence TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"
        ).ok();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS authored_sentence_recordings (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                sentence TEXT NOT NULL COLLATE NOCASE,
                take_no INTEGER NOT NULL,
                wav_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"
        ).ok();

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS rejected_suggestions (
                term TEXT PRIMARY KEY COLLATE NOCASE,
                rejected_at TEXT NOT NULL
            )"
        ).ok();

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS vocab_alt_spellings (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL COLLATE NOCASE,
                alt_spelling TEXT NOT NULL COLLATE NOCASE,
                created_at TEXT NOT NULL,
                UNIQUE(term, alt_spelling)
            )"
        ).ok();

        Ok(Db { conn })
    }

    // ==================== AUTHORED SENTENCES ====================

    /// Insert an authored sentence, linking it to ALL vocab terms it contains.
    /// The `primary_term` is the one that was prompted, but we also scan for others.
    pub fn insert_authored_sentence(&self, primary_term: &str, sentence: &str) -> Result<i64> {
        let now = now_str();
        let lower = sentence.to_lowercase();

        // Find all vocab terms present in this sentence
        let mut stmt = self.conn.prepare(
            "SELECT term FROM vocab WHERE reviewed = 1 AND spoken_override IS NOT NULL AND (curated IS NULL OR curated = 'kept')"
        )?;
        let all_terms: Vec<String> = stmt.query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut matched_terms: Vec<&str> = Vec::new();
        // Always include the primary term
        matched_terms.push(primary_term);
        // Add any other terms found in the sentence
        for t in &all_terms {
            if t.to_lowercase() != primary_term.to_lowercase() && lower.contains(&t.to_lowercase()) {
                matched_terms.push(t);
            }
        }
        matched_terms.dedup_by(|a, b| a.to_lowercase() == b.to_lowercase());

        let mut last_id = 0i64;
        for term in &matched_terms {
            self.conn.execute(
                "INSERT INTO authored_sentences (term, sentence, created_at) VALUES (?1, ?2, ?3)",
                params![term, sentence, now],
            )?;
            last_id = self.conn.last_insert_rowid();
        }
        Ok(last_id)
    }

    /// Get a vocab term that needs more sentences, weighted toward terms with fewer.
    /// Only considers terms that have overrides and are reviewed (usable for corpus).
    pub fn pick_term_for_authoring(&self) -> Result<Option<(VocabRow, i64)>> {
        // Count authored sentences per term, pick the one with fewest
        let mut stmt = self.conn.prepare(
            "SELECT v.id, v.term, v.spoken_auto, v.spoken_override, v.reviewed, v.description,
                    COALESCE(c.cnt, 0) as sentence_count
             FROM vocab v
             LEFT JOIN (SELECT term, COUNT(*) as cnt FROM authored_sentences GROUP BY term) c
                ON LOWER(c.term) = LOWER(v.term)
             WHERE v.reviewed = 1 AND v.spoken_override IS NOT NULL
                AND (v.curated IS NULL OR v.curated = 'kept')
                AND v.term NOT LIKE '%-%'
             ORDER BY sentence_count ASC, RANDOM()
             LIMIT 1"
        )?;
        let mut rows = stmt.query_map([], |row| {
            Ok((VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
                description: row.get(5)?,
            }, row.get::<_, i64>(6)?))
        })?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            _ => Ok(None),
        }
    }

    pub fn authored_sentence_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(DISTINCT sentence) FROM authored_sentences", [], |r| r.get(0))?)
    }

    pub fn authored_sentence_term_counts(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT term, COUNT(*) FROM authored_sentences GROUP BY term ORDER BY COUNT(*) DESC"
        )?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn list_authored_sentences(&self) -> Result<Vec<serde_json::Value>> {
        let mut stmt = self.conn.prepare(
            "SELECT a.id, a.term, a.sentence, a.created_at, COALESCE(r.cnt, 0) as recording_count
             FROM authored_sentences a
             LEFT JOIN (
                 SELECT LOWER(sentence) as sentence_key, COUNT(*) as cnt
                 FROM authored_sentence_recordings
                 GROUP BY LOWER(sentence)
             ) r ON r.sentence_key = LOWER(a.sentence)
             ORDER BY a.id DESC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, i64>(0)?,
                "term": row.get::<_, String>(1)?,
                "sentence": row.get::<_, String>(2)?,
                "created_at": row.get::<_, String>(3)?,
                "recording_count": row.get::<_, i64>(4)?,
            }))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn authored_sentence_recordings_for_sentence(&self, sentence: &str) -> Result<Vec<AuthoredSentenceRecordingRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, sentence, take_no, wav_path, created_at
             FROM authored_sentence_recordings
             WHERE LOWER(sentence) = LOWER(?1)
             ORDER BY take_no DESC, id DESC"
        )?;
        let rows = stmt.query_map(params![sentence], |row| {
            Ok(AuthoredSentenceRecordingRow {
                id: row.get(0)?,
                term: row.get(1)?,
                sentence: row.get(2)?,
                take_no: row.get(3)?,
                wav_path: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get all authored sentence recordings for eval. One row per take.
    pub fn authored_sentence_recordings_for_eval(&self) -> Result<Vec<AuthoredSentenceRecordingRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, sentence, take_no, wav_path, created_at
             FROM authored_sentence_recordings
             ORDER BY created_at ASC, id ASC"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(AuthoredSentenceRecordingRow {
                id: row.get(0)?,
                term: row.get(1)?,
                sentence: row.get(2)?,
                take_no: row.get(3)?,
                wav_path: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn authored_sentence_recordings_count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM authored_sentence_recordings", [], |r| r.get(0))?)
    }

    pub fn authored_sentences_with_recordings_count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(DISTINCT sentence) FROM authored_sentence_recordings",
            [],
            |r| r.get(0),
        )?)
    }

    pub fn add_authored_sentence_recording(&self, term: &str, sentence: &str, wav_path: &str) -> Result<i64> {
        let take_no = self.next_authored_sentence_recording_take_no(sentence)?;
        self.insert_authored_sentence_recording(term, sentence, take_no, wav_path)
    }

    pub fn insert_authored_sentence_recording(&self, term: &str, sentence: &str, take_no: i64, wav_path: &str) -> Result<i64> {
        let now = now_str();
        self.conn.execute(
            "INSERT INTO authored_sentence_recordings (term, sentence, take_no, wav_path, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![term, sentence, take_no, wav_path, now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn next_authored_sentence_recording_take_no(&self, sentence: &str) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COALESCE(MAX(take_no), 0) + 1 FROM authored_sentence_recordings WHERE LOWER(sentence) = LOWER(?1)",
            params![sentence],
            |r| r.get(0),
        )?)
    }

    pub fn get_authored_sentence_recording(&self, id: i64) -> Result<Option<AuthoredSentenceRecordingRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, sentence, take_no, wav_path, created_at
             FROM authored_sentence_recordings
             WHERE id = ?1"
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(AuthoredSentenceRecordingRow {
                id: row.get(0)?,
                term: row.get(1)?,
                sentence: row.get(2)?,
                take_no: row.get(3)?,
                wav_path: row.get(4)?,
                created_at: row.get(5)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn delete_authored_sentence_recording(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "DELETE FROM authored_sentence_recordings WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Get unique authored sentences for eval. Returns (sentence, primary_term).
    pub fn authored_sentences_for_eval(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT sentence, term FROM authored_sentences GROUP BY sentence ORDER BY id"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn update_authored_sentence(&self, id: i64, sentence: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE authored_sentences SET sentence = ?1 WHERE id = ?2",
            params![sentence, id],
        )?;
        Ok(())
    }

    pub fn get_authored_sentence(&self, id: i64) -> Result<Option<(String, String)>> {
        let mut stmt = self.conn.prepare("SELECT term, sentence FROM authored_sentences WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some((row.get(0)?, row.get(1)?)))
        } else {
            Ok(None)
        }
    }

    pub fn delete_authored_sentence(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM authored_sentences WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Get all authored sentences as plain text (for Markov chain building).
    pub fn all_authored_sentences(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT DISTINCT sentence FROM authored_sentences")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    // ==================== REJECTED SUGGESTIONS ====================

    pub fn reject_suggestion(&self, term: &str) -> Result<()> {
        let now = now_str();
        self.conn.execute(
            "INSERT OR IGNORE INTO rejected_suggestions (term, rejected_at) VALUES (?1, ?2)",
            params![term, now],
        )?;
        Ok(())
    }

    pub fn list_rejected_suggestions(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT term FROM rejected_suggestions")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
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
            "SELECT id, term, spoken_auto, spoken_override, reviewed, description FROM vocab {where_clause} ORDER BY {} LIMIT ?{idx} OFFSET ?{}",
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
                description: row.get(5)?,
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
    pub fn update_vocab_description(&self, id: i64, description: Option<&str>) -> Result<()> {
        self.conn.execute(
            "UPDATE vocab SET description = ?1, updated_at = ?2 WHERE id = ?3",
            params![description, now_str(), id],
        )?;
        Ok(())
    }

    pub fn delete_vocab_by_term(&self, term: &str) -> Result<()> {
        self.conn.execute("DELETE FROM vocab WHERE LOWER(term) = LOWER(?1)", params![term])?;
        // Clean up authored sentences linked to this term
        self.conn.execute("DELETE FROM authored_sentences WHERE LOWER(term) = LOWER(?1)", params![term])?;
        // Clean up rejected suggestions
        self.conn.execute("DELETE FROM rejected_suggestions WHERE LOWER(term) = LOWER(?1)", params![term])?;
        Ok(())
    }

    pub fn find_vocab_by_term(&self, term: &str) -> Result<Option<VocabRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, spoken_auto, spoken_override, reviewed, description FROM vocab WHERE LOWER(term) = LOWER(?1)"
        )?;
        let mut rows = stmt.query_map(params![term], |row| {
            Ok(VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
                description: row.get(5).ok().flatten(),
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

    /// Upsert a corpus pair. Dedup key is (term, original_lower, qwen_lower).
    /// On conflict: increment hit_count, update alignment data to latest.
    /// Returns (id, is_new) — is_new=true if this was a first sighting.
    pub fn upsert_corpus_pair(
        &self, term: &str, original: &str, qwen: &str, parakeet: &str,
        sentence: &str, spoken: &str,
        orig_align: Option<&str>, qwen_align: Option<&str>, parakeet_align: Option<&str>,
        cons_time: Option<&str>, trim_info: Option<&str>, is_mistake: bool,
        audio_ogg: Option<&[u8]>,
    ) -> Result<(i64, bool)> {
        let orig_lower = original.to_lowercase();
        let qwen_lower = qwen.to_lowercase();
        let now = now_str();

        // Check if this exact (term, orig_lower, qwen_lower) exists
        let existing: Option<i64> = self.conn.query_row(
            "SELECT id FROM corpus_pairs WHERE term = ?1 AND LOWER(original) = ?2 AND LOWER(qwen) = ?3",
            params![term, orig_lower, qwen_lower],
            |row| row.get(0),
        ).ok();

        if let Some(id) = existing {
            // Increment hit count, update alignment + audio to latest
            self.conn.execute(
                "UPDATE corpus_pairs SET hit_count = hit_count + 1, sentence = ?2, spoken = ?3, \
                 orig_alignment = ?4, qwen_alignment = ?5, parakeet_alignment = ?6, \
                 cons_time = ?7, trim_info = ?8, audio_ogg = ?9, updated_at = ?10 WHERE id = ?1",
                params![id, sentence, spoken, orig_align, qwen_align, parakeet_align, cons_time, trim_info, audio_ogg, now],
            )?;
            Ok((id, false))
        } else {
            self.conn.execute(
                "INSERT INTO corpus_pairs (term, original, qwen, parakeet, sentence, spoken, \
                 orig_alignment, qwen_alignment, parakeet_alignment, cons_time, trim_info, \
                 hit_count, is_mistake, audio_ogg, created_at, updated_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 1, ?12, ?13, ?14, ?14)",
                params![term, original, qwen, parakeet, sentence, spoken, orig_align, qwen_align,
                        parakeet_align, cons_time, trim_info, is_mistake as i32, audio_ogg, now],
            )?;
            Ok((self.conn.last_insert_rowid(), true))
        }
    }

    /// Get the next unreviewed mistake for the full-screen review flow.
    /// Returns the highest-hit-count unreviewed mistake.
    pub fn next_unreviewed_confusion(&self) -> Result<Option<serde_json::Value>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, original, qwen, parakeet, sentence, spoken, \
             orig_alignment, qwen_alignment, parakeet_alignment, cons_time, trim_info, \
             hit_count, audio_ogg \
             FROM corpus_pairs \
             WHERE is_mistake = 1 AND review_status IS NULL \
             ORDER BY hit_count DESC \
             LIMIT 1"
        )?;
        let row = stmt.query_row([], |row| {
            let orig_align: Option<String> = row.get(7)?;
            let qwen_align: Option<String> = row.get(8)?;
            let para_align: Option<String> = row.get(9)?;
            let cons_time: Option<String> = row.get(10)?;
            let trim_info: Option<String> = row.get(11)?;
            let audio: Option<Vec<u8>> = row.get(13)?;
            Ok(serde_json::json!({
                "id": row.get::<_, i64>(0)?,
                "term": row.get::<_, String>(1)?,
                "original": row.get::<_, String>(2)?,
                "qwen": row.get::<_, String>(3)?,
                "parakeet": row.get::<_, String>(4)?,
                "sentence": row.get::<_, String>(5)?,
                "spoken": row.get::<_, String>(6)?,
                "orig_alignment": orig_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "qwen_alignment": qwen_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "parakeet_alignment": para_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "cons_time": cons_time.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "trim_info": trim_info.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "hit_count": row.get::<_, i64>(12)?,
                "audio_b64": audio.map(|b| base64::engine::general_purpose::STANDARD.encode(&b)),
            }))
        });
        match row {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Progress for the confusions review: how many reviewed, how many total, how many remaining.
    pub fn confusions_review_progress(&self) -> Result<serde_json::Value> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM corpus_pairs WHERE is_mistake = 1", [], |r| r.get(0)
        )?;
        let reviewed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM corpus_pairs WHERE is_mistake = 1 AND review_status IS NOT NULL", [], |r| r.get(0)
        )?;
        let remaining: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM corpus_pairs WHERE is_mistake = 1 AND review_status IS NULL", [], |r| r.get(0)
        )?;
        Ok(serde_json::json!({"total": total, "reviewed": reviewed, "remaining": remaining}))
    }

    pub fn get_corpus_audio(&self, id: i64) -> Result<Option<Vec<u8>>> {
        Ok(self.conn.query_row(
            "SELECT audio_ogg FROM corpus_pairs WHERE id = ?1",
            params![id],
            |row| row.get(0),
        ).ok().flatten())
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

    /// Query corpus pairs with optional filtering.
    /// `review_filter`: None = all non-rejected, Some("unreviewed") = unreviewed mistakes only,
    /// Some("approved") = approved only, Some("rejected") = rejected only.
    pub fn corpus_pairs_query(&self, filter_term: Option<&str>, mistakes_only: bool, review_filter: Option<&str>, limit: usize, offset: usize) -> Result<Vec<serde_json::Value>> {
        let mut sql = String::from(
            "SELECT id, term, original, qwen, parakeet, sentence, spoken, orig_alignment, qwen_alignment, parakeet_alignment, cons_time, trim_info, hit_count, is_mistake, audio_ogg IS NOT NULL, review_status FROM corpus_pairs WHERE 1=1"
        );
        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(t) = filter_term {
            if !t.is_empty() {
                sql.push_str(&format!(" AND term = ?{param_idx}"));
                params_vec.push(Box::new(t.to_string()));
                param_idx += 1;
            }
        }
        if mistakes_only {
            sql.push_str(" AND is_mistake = 1");
        }
        match review_filter {
            Some("unreviewed") => sql.push_str(" AND review_status IS NULL AND is_mistake = 1"),
            Some("approved") => sql.push_str(" AND review_status = 'approved'"),
            Some("rejected") => sql.push_str(" AND review_status = 'rejected'"),
            _ => sql.push_str(" AND (review_status IS NULL OR review_status != 'rejected')"),
        }
        if mistakes_only || review_filter == Some("unreviewed") {
            sql.push_str(" ORDER BY hit_count DESC");
        } else {
            sql.push_str(" ORDER BY COALESCE(updated_at, created_at) DESC");
        }
        sql.push_str(&format!(" LIMIT ?{param_idx} OFFSET ?{}", param_idx + 1));
        params_vec.push(Box::new(limit as i64));
        params_vec.push(Box::new(offset as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            let orig_align: Option<String> = row.get(7)?;
            let qwen_align: Option<String> = row.get(8)?;
            let para_align: Option<String> = row.get(9)?;
            let cons_time: Option<String> = row.get(10)?;
            let trim_info: Option<String> = row.get(11)?;
            Ok(serde_json::json!({
                "id": row.get::<_, i64>(0)?,
                "term": row.get::<_, String>(1)?,
                "original": row.get::<_, String>(2)?,
                "qwen": row.get::<_, String>(3)?,
                "parakeet": row.get::<_, String>(4)?,
                "sentence": row.get::<_, String>(5)?,
                "spoken": row.get::<_, String>(6)?,
                "clean": true,
                "orig_alignment": orig_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "qwen_alignment": qwen_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "parakeet_alignment": para_align.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "cons_time": cons_time.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "trim_info": trim_info.and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()),
                "hit_count": row.get::<_, i64>(12)?,
                "is_mistake": row.get::<_, i64>(13)? != 0,
                "has_audio": row.get::<_, i64>(14)? != 0,
                "review_status": row.get::<_, Option<String>>(15)?,
            }))
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Summary stats for corpus: total pairs, mistakes, correct, per-term error rates.
    /// Excludes rejected pairs.
    pub fn corpus_stats(&self) -> Result<serde_json::Value> {
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM corpus_pairs WHERE review_status IS NULL OR review_status != 'rejected'", [], |r| r.get(0))?;
        let mistakes: i64 = self.conn.query_row("SELECT COUNT(*) FROM corpus_pairs WHERE is_mistake = 1 AND (review_status IS NULL OR review_status != 'rejected')", [], |r| r.get(0))?;
        let correct: i64 = self.conn.query_row("SELECT COUNT(*) FROM corpus_pairs WHERE is_mistake = 0 AND (review_status IS NULL OR review_status != 'rejected')", [], |r| r.get(0))?;
        let total_hits: i64 = self.conn.query_row("SELECT COALESCE(SUM(hit_count), 0) FROM corpus_pairs WHERE review_status IS NULL OR review_status != 'rejected'", [], |r| r.get(0))?;
        let unreviewed: i64 = self.conn.query_row("SELECT COUNT(*) FROM corpus_pairs WHERE is_mistake = 1 AND review_status IS NULL", [], |r| r.get(0))?;

        // Per-term stats: mistake_count, correct_count, total_hits
        let mut stmt = self.conn.prepare(
            "SELECT term, SUM(CASE WHEN is_mistake = 1 THEN 1 ELSE 0 END), \
             SUM(CASE WHEN is_mistake = 0 THEN 1 ELSE 0 END), \
             SUM(hit_count) FROM corpus_pairs WHERE review_status IS NULL OR review_status != 'rejected' GROUP BY term ORDER BY term COLLATE NOCASE"
        )?;
        let term_stats: Vec<serde_json::Value> = stmt.query_map([], |row| {
            Ok(serde_json::json!({
                "term": row.get::<_, String>(0)?,
                "mistakes": row.get::<_, i64>(1)?,
                "correct": row.get::<_, i64>(2)?,
                "hits": row.get::<_, i64>(3)?,
            }))
        })?.filter_map(|r| r.ok()).collect();

        Ok(serde_json::json!({
            "unique_pairs": total,
            "unique_mistakes": mistakes,
            "unique_correct": correct,
            "unreviewed_mistakes": unreviewed,
            "total_hits": total_hits,
            "terms": term_stats,
        }))
    }

    pub fn reset_corpus(&self) -> Result<()> {
        self.conn.execute("DELETE FROM corpus_pairs", [])?;
        Ok(())
    }

    /// Delete all corpus pairs for a specific term.
    pub fn delete_corpus_pairs_for_term(&self, term: &str) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM corpus_pairs WHERE LOWER(term) = LOWER(?1)",
            params![term],
        )?;
        Ok(n)
    }

    /// Mark a corpus pair as rejected (hidden from default mistakes view).
    pub fn reject_corpus_pair(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE corpus_pairs SET rejected = 1, review_status = 'rejected' WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    /// Mark a corpus pair as approved (confirmed real mistake, good training data).
    pub fn approve_corpus_pair(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE corpus_pairs SET review_status = 'approved' WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    // ==================== ALTERNATE SPELLINGS ====================

    /// Add an alternate acceptable spelling for a vocab term.
    /// Also retroactively marks existing corpus pairs as non-mistakes if they match.
    pub fn add_alt_spelling(&self, term: &str, alt_spelling: &str) -> Result<usize> {
        self.conn.execute(
            "INSERT OR IGNORE INTO vocab_alt_spellings (term, alt_spelling, created_at) VALUES (?1, ?2, ?3)",
            params![term, alt_spelling, now_str()],
        )?;
        // Retroactively mark matching corpus pairs as non-mistakes
        let updated = self.conn.execute(
            "UPDATE corpus_pairs SET is_mistake = 0 WHERE LOWER(term) = LOWER(?1) AND LOWER(qwen) = LOWER(?2)",
            params![term, alt_spelling],
        )?;
        Ok(updated)
    }

    /// Get all alternate spellings for a term.
    pub fn get_alt_spellings_for_term(&self, term: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT alt_spelling FROM vocab_alt_spellings WHERE LOWER(term) = LOWER(?1)"
        )?;
        let rows = stmt.query_map(params![term], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Get all alternate spellings as a map: term → set of acceptable spellings.
    /// The original term itself is included in each set.
    pub fn get_all_alt_spellings(&self) -> Result<HashMap<String, Vec<String>>> {
        let mut stmt = self.conn.prepare(
            "SELECT term, alt_spelling FROM vocab_alt_spellings"
        )?;
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (term, alt) = row?;
            map.entry(term.to_lowercase()).or_default().push(alt);
        }
        Ok(map)
    }

    /// Delete an alternate spelling.
    pub fn delete_alt_spelling(&self, term: &str, alt_spelling: &str) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM vocab_alt_spellings WHERE LOWER(term) = LOWER(?1) AND LOWER(alt_spelling) = LOWER(?2)",
            params![term, alt_spelling],
        )?;
        Ok(n)
    }

    /// Check if a qwen output matches the original or any alt spelling for a term.
    pub fn is_acceptable_spelling(&self, term: &str, original: &str, qwen: &str) -> Result<bool> {
        if original.to_lowercase() == qwen.to_lowercase() {
            return Ok(true);
        }
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vocab_alt_spellings WHERE LOWER(term) = LOWER(?1) AND LOWER(alt_spelling) = LOWER(?2)",
            params![term, qwen],
            |r| r.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get unique clean triplets for evaluation (deduplicated by original+qwen+parakeet).
    /// All corpus pairs for eval: (term, original, qwen, parakeet, is_mistake, hit_count).
    /// Each row is a unique (term, original, qwen) combination.
    pub fn corpus_eval_set(&self) -> Result<Vec<EvalItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT term, original, qwen, parakeet, is_mistake, hit_count FROM corpus_pairs ORDER BY term COLLATE NOCASE"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(EvalItem {
                term: row.get(0)?,
                original: row.get(1)?,
                qwen: row.get(2)?,
                parakeet: row.get(3)?,
                is_mistake: row.get::<_, i64>(4)? != 0,
                hit_count: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    /// Keep old method for backward compat
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
        let mut stmt = self.conn.prepare("SELECT text FROM sentences UNION SELECT text FROM candidate_sentences UNION SELECT sentence FROM authored_sentences")?;
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
        // Include both unreviewed terms AND reviewed terms missing overrides
        // curated can be NULL (not yet curated) or 'kept'
        let mut stmt = self.conn.prepare(
            "SELECT id FROM vocab WHERE (curated IS NULL OR curated = 'kept') AND (reviewed = 0 OR spoken_override IS NULL) ORDER BY id LIMIT ?1"
        )?;
        let rows = stmt.query_map(params![limit], |row| row.get(0))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub fn get_vocab(&self, id: i64) -> Result<Option<VocabRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, term, spoken_auto, spoken_override, reviewed, description FROM vocab WHERE id = ?1"
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
                description: row.get(5).ok().flatten(),
            })
        })?;
        match rows.next() {
            Some(r) => Ok(Some(r?)),
            None => Ok(None),
        }
    }

    /// All reviewed (approved) vocab terms for corpus generation.
    pub fn list_reviewed_vocab(&self) -> Result<Vec<VocabRow>> {
        // Only use terms that have been reviewed AND have a pronunciation override.
        // Skip hyphenated terms — they're compound names where each part is a normal word.
        let mut stmt = self.conn.prepare(
            "SELECT id, term, spoken_auto, spoken_override, reviewed, description FROM vocab WHERE reviewed = 1 AND spoken_override IS NOT NULL AND (curated IS NULL OR curated = 'kept') AND term NOT LIKE '%-%'"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(VocabRow {
                id: row.get(0)?, term: row.get(1)?, spoken_auto: row.get(2)?,
                spoken_override: row.get(3)?, reviewed: row.get::<_, i64>(4)? != 0,
                description: row.get(5).ok().flatten(),
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
        // "done" = reviewed AND has override (actually usable for corpus generation)
        // "to review" = missing override (regardless of reviewed flag or curated status)
        let done: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vocab WHERE (curated IS NULL OR curated = 'kept') AND reviewed = 1 AND spoken_override IS NOT NULL", [], |r| r.get(0))?;
        let to_review: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM vocab WHERE (curated IS NULL OR curated = 'kept') AND (reviewed = 0 OR spoken_override IS NULL)", [], |r| r.get(0))?;
        Ok((done, to_review, done + to_review))
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
