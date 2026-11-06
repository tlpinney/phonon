//! Session context: terms from the AI coding session the speaker is dictating into.
//!
//! When the target app is a terminal, editor or AI prompt box, the most recently written
//! Claude Code (`~/.claude/projects/*/*.jsonl`) or Codex (`~/.codex/sessions/YYYY/MM/DD/*.jsonl`)
//! transcript, if it changed in the last few minutes, is almost always the session being
//! talked to. Its recent messages hold the file names, identifiers and commands the speaker is
//! likely to say. Only identifier-shaped tokens are kept (separators, inner capitals, digits,
//! all caps), and only those the transcript plausibly says reach the correction prompt, as
//! `session_terms`. Nothing is written and nothing leaves the machine.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// A session written longer ago than this is not the one being dictated into.
pub const MAX_SESSION_AGE: Duration = Duration::from_secs(15 * 60);
/// Bytes read from the end of the transcript: the recent turns, not the whole session.
const TAIL_BYTES: u64 = 768 * 1024;
/// Characters kept from one message or tool result (tool output can be a whole file).
const MAX_BLOCK_CHARS: usize = 4_000;
/// Messages and tool calls read, newest last.
const MAX_BLOCKS: usize = 80;
/// Distinct candidate terms kept, most recent first.
const MAX_CANDIDATES: usize = 600;
/// Terms sent with one dictation.
pub const MAX_TERMS: usize = 12;

/// The transcript file of the session most recently written under `home`, if it changed
/// within `max_age` of `now`.
pub fn active_session(home: &Path, now: SystemTime, max_age: Duration) -> Option<PathBuf> {
    let mut newest: Option<(SystemTime, PathBuf)> = None;
    let mut consider = |path: PathBuf| {
        let Ok(modified) = fs::metadata(&path).and_then(|meta| meta.modified()) else {
            return;
        };
        let fresh = now
            .duration_since(modified)
            .map(|age| age <= max_age)
            .unwrap_or(true);
        if fresh && newest.as_ref().is_none_or(|(best, _)| modified > *best) {
            newest = Some((modified, path));
        }
    };
    for project in read_dirs(&home.join(".claude/projects")) {
        for file in read_files(&project, "jsonl") {
            consider(file);
        }
    }
    // Codex nests sessions by date: only the newest few day directories can be fresh.
    for day in newest_children(&home.join(".codex/sessions"), 3, 2) {
        for file in read_files(&day, "jsonl") {
            consider(file);
        }
    }
    newest.map(|(_, path)| path)
}

/// The recent message, tool call and tool result text of a Claude Code or Codex transcript.
pub fn session_text(path: &Path) -> String {
    let Ok(mut file) = fs::File::open(path) else {
        return String::new();
    };
    let length = file.metadata().map(|meta| meta.len()).unwrap_or(0);
    let start = length.saturating_sub(TAIL_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return String::new();
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut lines = text.lines();
    if start > 0 {
        lines.next(); // a partial line where the tail began
    }
    let mut blocks = Vec::new();
    for line in lines {
        let Ok(record) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        collect_blocks(&record, &mut blocks);
    }
    let skip = blocks.len().saturating_sub(MAX_BLOCKS);
    blocks.split_off(skip).join("\n")
}

/// Identifier-shaped tokens of `text`, most recent first, without repeats. A path also gives
/// its file name and the file name's stem, which is what a speaker says.
pub fn candidate_terms(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut terms = Vec::new();
    let tokens: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || "\"'`()[]{}<>,;|=*!?&$#^~\\".contains(c))
        .collect();
    for token in tokens.into_iter().rev() {
        let token = token.trim_matches(|c: char| ".:-/@".contains(c));
        for term in token_forms(token) {
            if seen.insert(term.to_lowercase()) {
                terms.push(term);
                if terms.len() >= MAX_CANDIDATES {
                    return terms;
                }
            }
        }
    }
    terms
}

/// The candidates the transcript plausibly says, at most `MAX_TERMS`, most recent first.
///
/// Stricter than screen context's matcher, which checks a handful of dictionary terms: a
/// session offers hundreds. A term is split into its spoken words (`session_context` and
/// `SessionContext` are "session context") and those words must appear in order in the
/// transcript, each the same or a near miss with the same first letter ("gimma" for gemma).
/// A one-word term must not be ordinary English; a longer one needs a word of five letters
/// or more, so `and_then` never matches "and then". A file name with its extension needs
/// "dot" after it. Terms that read the same are offered once.
pub fn matching_terms(transcript: &str, candidates: &[String]) -> Vec<String> {
    let spoken = crate::data::normalize(transcript);
    let said: Vec<&str> = spoken.split_whitespace().collect();
    let mut seen = std::collections::HashSet::new();
    let mut terms = Vec::new();
    for term in candidates {
        let (stem, extension) = match file_extension(term) {
            Some(extension) => (&term[..term.len() - extension.len() - 1], true),
            None => (term.as_str(), false),
        };
        let words = spoken_words(stem);
        if words.is_empty() {
            continue;
        }
        let by_words = worth_matching(&words);
        let compact: String = words.concat();
        let as_one = words.len() > 1
            && compact.len() >= 5
            && compact.chars().all(|c| c.is_ascii_alphabetic())
            && !crate::data::is_ordinary_english(&compact);
        if !by_words && !as_one {
            continue;
        }
        let key = (words.join(" "), extension);
        if seen.contains(&key) {
            continue;
        }
        if said_in_order(&said, &words, extension, by_words, as_one) {
            seen.insert(key);
            terms.push(term.clone());
            if terms.len() >= MAX_TERMS {
                break;
            }
        }
    }
    terms
}

/// The words a term is said as: split at separators, inner capitals and letter-digit
/// boundaries, lowercased. `prompt_lookup` -> [prompt, lookup]; `DraftProposer` -> [draft,
/// proposer]; `HTTPServer` -> [http, server]; `gpt5` -> [gpt, 5].
fn spoken_words(term: &str) -> Vec<String> {
    let chars: Vec<char> = term.chars().collect();
    let mut words = Vec::new();
    let mut current = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_ascii_alphanumeric() {
            if !current.is_empty() {
                words.push(std::mem::take(&mut current));
            }
            continue;
        }
        if let Some(&previous) = current.chars().last().as_ref() {
            let next_lower = chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
            let boundary = (previous.is_ascii_lowercase() && c.is_ascii_uppercase())
                || (previous.is_ascii_uppercase() && c.is_ascii_uppercase() && next_lower)
                || (previous.is_ascii_alphabetic() != c.is_ascii_alphabetic());
            if boundary {
                words.push(std::mem::take(&mut current));
            }
        }
        current.push(c.to_ascii_lowercase());
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

fn worth_matching(words: &[String]) -> bool {
    if words.len() == 1 {
        let word = &words[0];
        return word.len() >= 4
            && word.chars().all(|c| c.is_ascii_alphabetic())
            && !crate::data::is_ordinary_english(word);
    }
    words.iter().any(|word| word.len() >= 5 && word.chars().all(|c| c.is_ascii_alphabetic()))
}

/// Whether `words` occur consecutively in `said` (each the same or a near miss), followed by
/// "dot" when the term is a file name with its extension. The term said as one word, exactly
/// (`github` for `GitHub`), counts too.
fn said_in_order(
    said: &[&str],
    words: &[String],
    extension: bool,
    by_words: bool,
    as_one: bool,
) -> bool {
    let same = |spoken: &str, word: &str| {
        spoken == word || crate::data::plausibly_same_word(spoken, word)
    };
    let followed_ok = |end: usize| !extension || said.get(end).is_some_and(|w| *w == "dot");
    for start in 0..said.len() {
        // word by word
        if by_words
            && start + words.len() <= said.len()
            && words.iter().zip(&said[start..]).all(|(word, spoken)| same(spoken, word))
            && followed_ok(start + words.len())
        {
            return true;
        }
        // the whole term said as one word, exactly (`github` for `GitHub`)
        let compact: String = words.concat();
        if as_one && said[start] == compact && followed_ok(start + 1) {
            return true;
        }
    }
    false
}

fn file_extension(term: &str) -> Option<String> {
    let (stem, extension) = term.rsplit_once('.')?;
    ((1..=5).contains(&extension.len())
        && !stem.is_empty()
        && extension.chars().all(|c| c.is_ascii_alphabetic()))
    .then(|| extension.to_string())
}

/// The session dictionary: the active session's candidate terms, kept in memory and read
/// again only when the session file changes (path, size or modification time). Never
/// written anywhere; a new session replaces it.
#[derive(Debug, Default)]
pub struct SessionDictionary {
    key: Option<(PathBuf, SystemTime, u64)>,
    candidates: Vec<String>,
    /// Rebuilt since it was last saved.
    unsaved: bool,
    /// Transcript reads and parses (for tests and diagnostics).
    pub reads: usize,
}

/// One saved session dictionary (`session-dictionaries/<tool>-<session>.json`): the terms of
/// one AI coding session, most recent first, for training and review.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SavedSessionDictionary {
    pub schema_version: u32,
    /// `claude-code` or `codex`.
    pub tool: String,
    pub session_file: String,
    pub updated_at_unix_ms: u128,
    pub terms: Vec<String>,
}

impl SessionDictionary {
    /// The session terms for one dictation, or none when no session is active.
    pub fn terms(&mut self, home: &Path, transcript: &str, now: SystemTime) -> Vec<String> {
        let Some(path) = active_session(home, now, MAX_SESSION_AGE) else {
            self.key = None;
            self.candidates.clear();
            return Vec::new();
        };
        let Ok(meta) = fs::metadata(&path) else {
            return Vec::new();
        };
        let key = (
            path.clone(),
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            meta.len(),
        );
        if self.key.as_ref() != Some(&key) {
            self.candidates = candidate_terms(&session_text(&path));
            self.key = Some(key);
            self.unsaved = true;
            self.reads += 1;
        }
        matching_terms(transcript, &self.candidates)
    }

    /// Write the current session dictionary into `dir` when it changed since the last save,
    /// and return its file name; None when no session is active.
    pub fn save(&mut self, dir: &Path) -> std::io::Result<Option<String>> {
        let Some((path, _, _)) = &self.key else {
            return Ok(None);
        };
        let tool = if path.components().any(|part| part.as_os_str() == ".codex") {
            "codex"
        } else {
            "claude-code"
        };
        let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("session");
        let name = format!("{tool}-{stem}.json");
        if self.unsaved {
            fs::create_dir_all(dir)?;
            let saved = SavedSessionDictionary {
                schema_version: 1,
                tool: tool.to_string(),
                session_file: path.display().to_string(),
                updated_at_unix_ms: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_millis())
                    .unwrap_or(0),
                terms: self.candidates.clone(),
            };
            let target = dir.join(&name);
            let staging = dir.join(format!(".{name}.tmp"));
            fs::write(&staging, serde_json::to_vec_pretty(&saved)?)?;
            fs::rename(&staging, &target)?;
            self.unsaved = false;
        }
        Ok(Some(name))
    }
}

/// How long a dictation pasted into a session waits for the message the speaker sends.
pub const SEND_WAIT: Duration = Duration::from_secs(10 * 60);
/// Share of the pasted words a sent message must contain to be that dictation, edited.
const SENT_OVERLAP: f64 = 0.5;

/// A dictation pasted into an AI coding session, waiting for the speaker to send it. The
/// session transcript then holds the message as sent, after any fixes: the edit Phonon cannot
/// read from a terminal's screen.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingSend {
    /// The dictation's request id (its recording and the bar's suggestion queue use it).
    pub id: String,
    pub pasted: String,
    pub session: PathBuf,
    /// Session transcript length when the dictation was pasted: the send comes after it.
    pub offset: u64,
    pub since: SystemTime,
    /// Local history is on: keep the edit in the recording.
    pub record: bool,
}

/// Dictations waiting for their send, one per session (the latest: a later pass of the same
/// dictation replaces an earlier one).
#[derive(Debug, Default)]
pub struct PendingSends {
    pending: Vec<PendingSend>,
    /// (session, length) last read, so an unchanged transcript is not read again.
    read_at: Vec<(PathBuf, u64)>,
}

impl PendingSends {
    pub fn add(&mut self, send: PendingSend) {
        self.pending.retain(|other| other.session != send.session);
        self.pending.push(send);
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Each waiting dictation whose send has appeared, with the message as sent; dictations
    /// whose next message is something else, or that waited too long, are dropped.
    pub fn take_sent(&mut self, now: SystemTime) -> Vec<(PendingSend, String)> {
        let mut sent = Vec::new();
        let mut keep = Vec::new();
        for send in std::mem::take(&mut self.pending) {
            let expired = now
                .duration_since(send.since)
                .is_ok_and(|waited| waited > SEND_WAIT);
            let length = fs::metadata(&send.session).map(|meta| meta.len()).unwrap_or(0);
            let unchanged = self
                .read_at
                .iter()
                .any(|(path, read)| *path == send.session && *read == length);
            if unchanged || length <= send.offset {
                if !expired {
                    keep.push(send);
                }
                continue;
            }
            self.read_at.retain(|(path, _)| *path != send.session);
            self.read_at.push((send.session.clone(), length));
            let (messages, _) = user_messages_since(&send.session, send.offset);
            match messages.first() {
                Some(message) if is_sent_version(&send.pasted, message) => {
                    let message = message.clone();
                    sent.push((send, message));
                }
                Some(_) => {} // the next message is something else: the dictation was not sent
                None if !expired => keep.push(send),
                None => {}
            }
        }
        self.pending = keep;
        sent
    }
}

/// The messages the user sent in a Claude Code or Codex transcript after byte `offset`
/// (complete lines only), and the offset after the last complete line.
pub fn user_messages_since(path: &Path, offset: u64) -> (Vec<String>, u64) {
    let Ok(mut file) = fs::File::open(path) else {
        return (Vec::new(), offset);
    };
    if file.seek(SeekFrom::Start(offset)).is_err() {
        return (Vec::new(), offset);
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return (Vec::new(), offset);
    }
    let complete = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |end| end + 1);
    let text = String::from_utf8_lossy(&bytes[..complete]);
    let messages = text
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|record| user_message(&record))
        .collect();
    (messages, offset + complete as u64)
}

/// The text of a message the user typed or dictated, if this record is one. Claude Code
/// writes a prompt as a `user` record with string content, and one sent while the agent works
/// as a `queue-operation` enqueue; Codex as a user `message` response item. Injected context
/// (`<environment_context>`, `<command-name>` and the like) is not the user's.
fn user_message(record: &serde_json::Value) -> Option<String> {
    let kind = record.get("type").and_then(|kind| kind.as_str())?;
    let text = match kind {
        "user" => record.get("message")?.get("content")?.as_str()?.to_string(),
        "queue-operation" => {
            if record.get("operation").and_then(|op| op.as_str()) != Some("enqueue") {
                return None;
            }
            record.get("content")?.as_str()?.to_string()
        }
        "response_item" => {
            let payload = record.get("payload")?;
            if payload.get("type").and_then(|t| t.as_str()) != Some("message")
                || payload.get("role").and_then(|r| r.as_str()) != Some("user")
            {
                return None;
            }
            payload
                .get("content")?
                .as_array()?
                .iter()
                .filter(|item| item.get("type").and_then(|t| t.as_str()) == Some("input_text"))
                .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => return None,
    };
    let text = text.trim();
    (!text.is_empty() && !text.starts_with('<')).then(|| text.to_string())
}

/// Whether `message` is the pasted dictation as sent: it holds at least half of its words.
fn is_sent_version(pasted: &str, message: &str) -> bool {
    let pasted = crate::data::normalize(pasted);
    let pasted: Vec<&str> = pasted.split_whitespace().collect();
    if pasted.is_empty() {
        return false;
    }
    let message = crate::data::normalize(message);
    let mut available: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for word in message.split_whitespace() {
        *available.entry(word).or_default() += 1;
    }
    let mut found = 0;
    for word in &pasted {
        if let Some(count) = available.get_mut(word) {
            if *count > 0 {
                *count -= 1;
                found += 1;
            }
        }
    }
    found as f64 / pasted.len() as f64 >= SENT_OVERLAP
}

fn token_forms(token: &str) -> Vec<String> {
    if token.contains("://") || token.starts_with("www.") {
        return Vec::new();
    }
    let mut forms = Vec::new();
    let name = token.rsplit('/').next().unwrap_or(token);
    if token.contains('/') && is_identifier(name) {
        forms.push(name.to_string());
    } else if is_identifier(token) {
        forms.push(token.to_string());
    }
    // `prompt_lookup.py` is also said as `prompt lookup`.
    if let Some((stem, extension)) = name.rsplit_once('.') {
        if (1..=5).contains(&extension.len())
            && extension.chars().all(|c| c.is_ascii_alphanumeric())
            && is_identifier(stem)
        {
            forms.push(stem.to_string());
        }
    }
    forms
}

/// A token worth hinting: 3 to 48 characters, mostly letters, shaped like code or a name
/// rather than an ordinary word (an inner separator or capital, a digit, or all capitals).
fn is_identifier(token: &str) -> bool {
    let length = token.chars().count();
    if !(3..=48).contains(&length) {
        return false;
    }
    if !token.chars().all(|c| c.is_ascii_alphanumeric() || "_-.+".contains(c)) {
        return false;
    }
    let letters = token.chars().filter(|c| c.is_ascii_alphabetic()).count();
    // Mostly letters, or a letter first (`H100`, `B200`); not `13` or `2026-11-06`.
    let leading_letter = token.starts_with(|c: char| c.is_ascii_alphabetic());
    if letters == 0 || (letters * 2 < length && !leading_letter) {
        return false;
    }
    // Commit ids and other hex runs.
    if length >= 7 && token.chars().all(|c| c.is_ascii_hexdigit()) {
        return false;
    }
    let inner = &token[1..];
    let separator = inner.contains(['_', '-', '.', '+']);
    let camel = token
        .as_bytes()
        .windows(2)
        .any(|pair| pair[0].is_ascii_lowercase() && pair[1].is_ascii_uppercase());
    let digit = token.chars().any(|c| c.is_ascii_digit());
    let caps = letters >= 2 && token.chars().all(|c| !c.is_ascii_lowercase());
    separator || camel || digit || caps
}

fn collect_blocks(record: &serde_json::Value, blocks: &mut Vec<String>) {
    // Claude Code: {"type":"user"|"assistant","message":{"content": str | [items]}}
    if let Some(message) = record.get("message") {
        if matches!(
            record.get("type").and_then(|kind| kind.as_str()),
            Some("user" | "assistant")
        ) {
            collect_content(message.get("content"), blocks);
        }
        return;
    }
    // Codex: {"type":"response_item","payload":{...}}
    if record.get("type").and_then(|kind| kind.as_str()) != Some("response_item") {
        return;
    }
    let Some(payload) = record.get("payload") else {
        return;
    };
    match payload.get("type").and_then(|kind| kind.as_str()) {
        Some("message") => {
            if matches!(
                payload.get("role").and_then(|role| role.as_str()),
                Some("user" | "assistant")
            ) {
                collect_content(payload.get("content"), blocks);
            }
        }
        Some("custom_tool_call") => push(blocks, payload.get("input")),
        Some("function_call") => push(blocks, payload.get("arguments")),
        Some("custom_tool_call_output" | "function_call_output") => {
            collect_content(payload.get("output"), blocks)
        }
        _ => {}
    }
}

fn collect_content(content: Option<&serde_json::Value>, blocks: &mut Vec<String>) {
    match content {
        Some(serde_json::Value::String(_)) => push(blocks, content),
        Some(serde_json::Value::Array(items)) => {
            for item in items {
                match item.get("type").and_then(|kind| kind.as_str()) {
                    Some("text" | "input_text" | "output_text") => push(blocks, item.get("text")),
                    Some("tool_use") => {
                        if let Some(input) = item.get("input") {
                            let mut text = String::new();
                            json_strings(input, &mut text);
                            push_text(blocks, &text);
                        }
                    }
                    Some("tool_result") => collect_content(item.get("content"), blocks),
                    _ => {} // thinking, images, reasoning
                }
            }
        }
        _ => {}
    }
}

fn json_strings(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::String(text) => {
            out.push_str(text);
            out.push('\n');
        }
        serde_json::Value::Array(items) => items.iter().for_each(|item| json_strings(item, out)),
        serde_json::Value::Object(map) => map.values().for_each(|item| json_strings(item, out)),
        _ => {}
    }
}

fn push(blocks: &mut Vec<String>, value: Option<&serde_json::Value>) {
    if let Some(text) = value.and_then(|value| value.as_str()) {
        push_text(blocks, text);
    }
}

fn push_text(blocks: &mut Vec<String>, text: &str) {
    if text.trim().is_empty() {
        return;
    }
    // Keep the end of a long block: a tool's last lines (an error, a summary) matter most.
    let count = text.chars().count();
    let text = if count > MAX_BLOCK_CHARS {
        text.chars().skip(count - MAX_BLOCK_CHARS).collect()
    } else {
        text.to_string()
    };
    blocks.push(text);
}

fn read_dirs(path: &Path) -> Vec<PathBuf> {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.is_dir())
                .collect()
        })
        .unwrap_or_default()
}

fn read_files(path: &Path, extension: &str) -> Vec<PathBuf> {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.extension().is_some_and(|ext| ext == extension))
                .collect()
        })
        .unwrap_or_default()
}

/// The `keep` lexically greatest directories `depth` levels below `root` (Codex's
/// `YYYY/MM/DD`: greatest is newest), searching only the greatest at each level above.
fn newest_children(root: &Path, depth: usize, keep: usize) -> Vec<PathBuf> {
    let mut level = vec![root.to_path_buf()];
    for step in 0..depth {
        let mut next: Vec<PathBuf> = level.iter().flat_map(|dir| read_dirs(dir)).collect();
        next.sort();
        let take = if step + 1 == depth { keep } else { 2 };
        level = next.into_iter().rev().take(take).collect();
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_kept_and_ordinary_words_are_not() {
        let text = "Edit sidecar/prompt_lookup.py and DraftProposer, then run cargo test \
                    with CUDA 13 on the H100; see https://example.com/x and commit 0ac902a1.";
        let terms = candidate_terms(text);
        for wanted in ["prompt_lookup.py", "prompt_lookup", "DraftProposer", "CUDA", "H100"] {
            assert!(terms.contains(&wanted.to_string()), "{wanted} in {terms:?}");
        }
        for unwanted in ["Edit", "cargo", "test", "example.com", "0ac902a1", "sidecar"] {
            assert!(!terms.contains(&unwanted.to_string()), "{unwanted} in {terms:?}");
        }
    }

    #[test]
    fn the_most_recent_mention_comes_first() {
        let terms = candidate_terms("old_name first, then new_name");
        assert_eq!(terms, vec!["new_name", "old_name"]);
    }

    #[test]
    fn spoken_identifiers_match_and_unrelated_ones_do_not() {
        let candidates = candidate_terms("prompt_lookup.py DraftProposer stream_refiner MLX");
        let terms = matching_terms("open the prompt lookup file and the draft proposer", &candidates);
        assert_eq!(terms, vec!["DraftProposer", "prompt_lookup"]);
        assert!(matching_terms("let's get lunch", &candidates).is_empty());
        // ordinary phrases never match helper names; one-word English terms never match
        let helpers = candidate_terms("and_then as_ref then_some FAILED AGENTS promptId GitHub");
        assert!(matching_terms("and then I failed the agents prompt lookup", &helpers).is_empty());
        assert_eq!(matching_terms("push it to github", &helpers), vec!["GitHub"]);
        // a near miss of a long word still matches
        let models = candidate_terms("gemma_loader");
        assert_eq!(matching_terms("fix the gimma loader", &models), vec!["gemma_loader"]);
        // the extension said: the file name itself
        assert_eq!(
            matching_terms("open prompt lookup dot pie", &candidates),
            vec!["prompt_lookup.py", "prompt_lookup"]
        );
    }

    #[test]
    fn claude_and_codex_transcripts_are_read() {
        let dir = std::env::temp_dir().join(format!("phonon-session-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let claude = dir.join("claude.jsonl");
        fs::write(
            &claude,
            [
                r#"{"type":"user","message":{"role":"user","content":"rename build_envelope"}}"#,
                r#"{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"secret_plan"},{"type":"tool_use","input":{"file_path":"/x/crates/session_context.rs"}}]}}"#,
                r#"{"type":"user","message":{"content":[{"type":"tool_result","content":"error in parse_record"}]}}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let terms = candidate_terms(&session_text(&claude));
        for wanted in ["build_envelope", "session_context.rs", "session_context", "parse_record"] {
            assert!(terms.contains(&wanted.to_string()), "{wanted} in {terms:?}");
        }
        assert!(!terms.contains(&"secret_plan".to_string()), "thinking is not read");

        let codex = dir.join("codex.jsonl");
        fs::write(
            &codex,
            [
                r#"{"type":"response_item","payload":{"type":"message","role":"developer","content":[{"type":"input_text","text":"system_prompt_term"}]}}"#,
                r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"fix the TokenCache"}]}}"#,
                r#"{"type":"response_item","payload":{"type":"custom_tool_call","input":"rg load_weights"}}"#,
            ]
            .join("\n"),
        )
        .unwrap();
        let terms = candidate_terms(&session_text(&codex));
        assert!(terms.contains(&"TokenCache".to_string()));
        assert!(terms.contains(&"load_weights".to_string()));
        assert!(!terms.contains(&"system_prompt_term".to_string()), "developer text is not read");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn only_a_fresh_session_is_active() {
        let home = std::env::temp_dir().join(format!("phonon-home-{}", std::process::id()));
        let project = home.join(".claude/projects/-x");
        let day = home.join(".codex/sessions/2026/11/06");
        fs::create_dir_all(&project).unwrap();
        fs::create_dir_all(&day).unwrap();
        fs::write(project.join("a.jsonl"), "{}").unwrap();
        std::thread::sleep(Duration::from_millis(20));
        fs::write(day.join("rollout.jsonl"), "{}").unwrap();
        let now = SystemTime::now();
        assert_eq!(
            active_session(&home, now, MAX_SESSION_AGE),
            Some(day.join("rollout.jsonl")),
            "the newest of both tools"
        );
        let later = now + Duration::from_secs(3600);
        assert_eq!(active_session(&home, later, MAX_SESSION_AGE), None, "stale: none");
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_pasted_dictation_is_matched_to_the_message_sent() {
        let dir = std::env::temp_dir().join(format!("phonon-send-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let session = dir.join("s.jsonl");
        fs::write(&session, "{\"type\":\"user\",\"message\":{\"content\":\"earlier\"}}\n").unwrap();
        let offset = fs::metadata(&session).unwrap().len();
        let now = SystemTime::now();
        let mut sends = PendingSends::default();
        sends.add(PendingSend {
            id: "p1".into(),
            pasted: "make the edit corrections work in clawed code and codex".into(),
            session: session.clone(),
            offset,
            since: now,
            record: true,
        });
        assert!(sends.take_sent(now).is_empty(), "nothing sent yet");
        let mut file = fs::OpenOptions::new().append(true).open(&session).unwrap();
        use std::io::Write;
        // the agent's own lines, then the user's fixed message
        writeln!(file, r#"{{"type":"assistant","message":{{"content":[{{"type":"text","text":"working"}}]}}}}"#).unwrap();
        writeln!(file, r#"{{"type":"user","message":{{"content":"make the edit corrections work in Claude Code and Codex"}}}}"#).unwrap();
        drop(file);
        let sent = sends.take_sent(now);
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0.id, "p1");
        assert_eq!(sent[0].1, "make the edit corrections work in Claude Code and Codex");
        assert!(sends.is_empty(), "matched: no longer waiting");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn queued_and_codex_messages_count_and_injected_context_does_not() {
        let records = [
            r#"{"type":"queue-operation","operation":"enqueue","content":"also fix the gemma loader"}"#,
            r#"{"type":"queue-operation","operation":"remove","content":"also fix the gemma loader"}"#,
            r#"{"type":"user","message":{"content":"<command-name>/clear</command-name>"}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","content":"ok"}]}}"#,
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"<environment_context>x</environment_context>"}]}}"#,
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"rerun the token cache tests"}]}}"#,
        ];
        let messages: Vec<String> = records
            .iter()
            .filter_map(|line| user_message(&serde_json::from_str(line).unwrap()))
            .collect();
        assert_eq!(messages, vec!["also fix the gemma loader", "rerun the token cache tests"]);
    }

    #[test]
    fn a_different_next_message_drops_the_dictation() {
        assert!(is_sent_version("rename the draft proposer", "rename the DraftProposer"));
        assert!(!is_sent_version("rename the draft proposer", "never mind, let's eat"));
    }

    #[test]
    fn the_session_dictionary_is_read_again_only_when_the_session_changes() {
        let home = std::env::temp_dir().join(format!("phonon-dict-{}", std::process::id()));
        let project = home.join(".claude/projects/-x");
        fs::create_dir_all(&project).unwrap();
        let session = project.join("s.jsonl");
        fs::write(
            &session,
            r#"{"type":"user","message":{"content":"look at load_weights"}}"#,
        )
        .unwrap();
        let mut dictionary = SessionDictionary::default();
        let now = SystemTime::now();
        assert_eq!(dictionary.terms(&home, "check load weights", now), vec!["load_weights"]);
        assert_eq!(dictionary.terms(&home, "and load weights again", now), vec!["load_weights"]);
        assert_eq!(dictionary.reads, 1, "unchanged session: one read");
        fs::write(
            &session,
            "{\"type\":\"user\",\"message\":{\"content\":\"look at load_weights\"}}\n\
             {\"type\":\"user\",\"message\":{\"content\":\"now save_checkpoint\"}}",
        )
        .unwrap();
        assert_eq!(dictionary.terms(&home, "call save checkpoint", now), vec!["save_checkpoint"]);
        assert_eq!(dictionary.reads, 2, "changed session: read again");
        let out = home.join("saved");
        assert_eq!(dictionary.save(&out).unwrap().as_deref(), Some("claude-code-s.json"));
        let saved: SavedSessionDictionary =
            serde_json::from_slice(&fs::read(out.join("claude-code-s.json")).unwrap()).unwrap();
        assert_eq!(saved.tool, "claude-code");
        assert_eq!(saved.terms, vec!["save_checkpoint", "load_weights"]);
        let later = now + Duration::from_secs(3600);
        assert!(dictionary.terms(&home, "call save checkpoint", later).is_empty());
        fs::remove_dir_all(&home).unwrap();
    }
}

#[cfg(test)]
mod edit_tests {
    #[test]
    fn a_fix_made_before_sending_becomes_a_term_suggestion() {
        let request = crate::edits::ObservedEdit {
            pasted: "Let's make the edit corrections work in clawed code and codex.".into(),
            edited: "Let's make the edit corrections work in Claude Code and Codex.".into(),
            end_reason: "session_message".into(),
            ..Default::default()
        };
        let analysis = request.analyze(&crate::data::DictionaryFile::default());
        let event = crate::edits::observed_event(Some("p1"), &analysis, false);
        let text = event.to_string();
        assert!(text.contains("Claude Code"), "{text}");
    }
}
