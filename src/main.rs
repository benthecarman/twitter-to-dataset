use clap::{Parser, ValueEnum};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Semaphore;

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "twitter-to-dataset",
    about = "Convert Twitter archive to Alpaca fine-tune dataset"
)]
struct Args {
    /// Path to a Twitter archive directory, data directory, tweets.js, or note-tweet.js
    #[arg(long)]
    archive: PathBuf,

    /// Output JSONL file
    #[arg(long, default_value = "dataset.jsonl")]
    output: PathBuf,

    /// Output JSONL file for direct messages
    #[arg(long, default_value = "dms-dataset.jsonl")]
    dms_output: PathBuf,

    /// Also generate a separate dataset from outbound DMs
    #[arg(long)]
    include_dms: bool,

    /// Generate only the DM dataset
    #[arg(long)]
    dms_only: bool,

    /// Owner account id for outbound DM detection; read from account.js when omitted
    #[arg(long)]
    owner_id: Option<String>,

    /// Inference backend to use
    #[arg(long, value_enum, default_value_t = BackendKind::Ollama)]
    backend: BackendKind,

    /// Model to use for instruction generation
    #[arg(long, default_value = "qwen3:14b")]
    model: String,

    /// Ollama base URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// OpenAI base URL
    #[arg(long, default_value = "https://api.openai.com")]
    openai_base_url: String,

    /// OpenAI API key; can also be set with OPENAI_API_KEY
    #[arg(long, env = "OPENAI_API_KEY")]
    api_key: Option<String>,

    /// Number of concurrent backend requests; defaults to 1 for Ollama, 4 for OpenAI
    #[arg(long)]
    workers: Option<usize>,

    /// Timeout for each backend request in seconds
    #[arg(long, default_value_t = 30)]
    timeout_secs: u64,

    /// Minimum tweet length after cleaning (chars)
    #[arg(long, default_value_t = 30)]
    min_length: usize,

    /// Exclude replies instead of cleaning their leading @mentions
    #[arg(long, conflicts_with = "only_replies")]
    exclude_replies: bool,

    /// Include only replies
    #[arg(long, conflicts_with = "exclude_replies")]
    only_replies: bool,

    /// Only process the first N kept records
    #[arg(long)]
    limit: Option<usize>,

    /// Print kept examples and exit without calling a model backend or writing output
    #[arg(long)]
    dry_run: bool,
}

// ── Types ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendKind {
    Ollama,
    Openai,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Ollama => f.write_str("ollama"),
            BackendKind::Openai => f.write_str("openai"),
        }
    }
}

#[derive(Clone, Debug)]
struct BackendConfig {
    kind: BackendKind,
    model: String,
    base_url: String,
    api_key: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct AlpacaRecord {
    instruction: String,
    input: String,
    output: String,
}

#[derive(Deserialize, Debug)]
struct OllamaResponse {
    message: OllamaMessage,
    done_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OllamaMessage {
    content: String,
    thinking: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Deserialize, Debug)]
struct OpenAiMessage {
    content: Option<String>,
}

// ── Tweet parsing ──────────────────────────────────────────────────────────────

#[derive(Debug)]
struct TweetText {
    source: &'static str,
    id: Option<String>,
    text: String,
    is_retweet: bool,
    is_reply: bool,
}

#[derive(Debug, Clone)]
struct FilteredTweet {
    source: &'static str,
    id: Option<String>,
    text: String,
    is_reply: bool,
}

fn archive_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }

    let data_dir = if path.file_name().and_then(|s| s.to_str()) == Some("data") {
        path.to_path_buf()
    } else {
        path.join("data")
    };

    ["tweets.js", "note-tweet.js"]
        .into_iter()
        .map(|name| data_dir.join(name))
        .filter(|candidate| candidate.exists())
        .collect()
}

fn archive_data_dir(path: &Path) -> PathBuf {
    if path.is_file() {
        return path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
    }

    if path.file_name().and_then(|s| s.to_str()) == Some("data") {
        path.to_path_buf()
    } else {
        path.join("data")
    }
}

fn dm_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }

    let data_dir = archive_data_dir(path);
    ["direct-messages.js", "direct-messages-group.js"]
        .into_iter()
        .map(|name| data_dir.join(name))
        .filter(|candidate| candidate.exists())
        .collect()
}

fn parse_archive_json(path: &Path) -> anyhow::Result<Vec<Value>> {
    let raw = std::fs::read_to_string(path)?;

    // Strip JS assignment: window.YTD.<name>.part0 = [...]
    let start = raw
        .find('[')
        .ok_or_else(|| anyhow::anyhow!("Could not find JSON array in {}", path.display()))?;
    let json_str = &raw[start..];

    serde_json::from_str(json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", path.display(), e))
}

fn load_owner_id(path: &Path, override_id: Option<&str>) -> anyhow::Result<String> {
    if let Some(owner_id) = override_id {
        return Ok(owner_id.to_string());
    }

    let account_path = archive_data_dir(path).join("account.js");
    let data = parse_archive_json(&account_path)?;
    data.iter()
        .find_map(|value| {
            value
                .get("account")
                .and_then(|account| account.get("accountId"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .ok_or_else(|| anyhow::anyhow!("Could not find accountId in {}", account_path.display()))
}

fn push_tweet_from_value(value: &Value, tweets: &mut Vec<TweetText>) {
    if let Some(tweet) = value.get("tweet").or_else(|| {
        if value.get("full_text").is_some() {
            Some(value)
        } else {
            None
        }
    }) {
        if let Some(text) = tweet.get("full_text").and_then(Value::as_str) {
            tweets.push(TweetText {
                source: "tweets.js",
                id: tweet
                    .get("id_str")
                    .or_else(|| tweet.get("id"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                text: text.to_string(),
                is_retweet: tweet
                    .get("retweeted")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    || text.starts_with("RT @"),
                is_reply: tweet.get("in_reply_to_status_id_str").is_some()
                    || tweet.get("in_reply_to_status_id").is_some(),
            });
        }
    } else if let Some(note_tweet) = value.get("noteTweet") {
        if let Some(text) = note_tweet
            .get("core")
            .and_then(|core| core.get("text"))
            .and_then(Value::as_str)
        {
            tweets.push(TweetText {
                source: "note-tweet.js",
                id: note_tweet
                    .get("noteTweetId")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                text: text.to_string(),
                is_retweet: false,
                is_reply: false,
            });
        }
    }
}

fn load_filtered_tweets(
    path: &Path,
    min_length: usize,
    exclude_replies: bool,
    only_replies: bool,
    limit: Option<usize>,
) -> anyhow::Result<(Vec<FilteredTweet>, usize)> {
    let files = archive_files(path);
    if files.is_empty() {
        anyhow::bail!(
            "Could not find tweets.js or note-tweet.js under {}",
            path.display()
        );
    }

    let mut raw_count = 0;
    let mut seen_cleaned = HashSet::new();
    let mut filtered = Vec::new();

    for file in files {
        let data = parse_archive_json(&file)?;
        let source_name = file.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let before_raw = raw_count;
        let before_kept = filtered.len();

        for value in data {
            let mut tweets = Vec::new();
            push_tweet_from_value(&value, &mut tweets);

            for tweet in tweets {
                raw_count += 1;
                let cleaned = clean_text(&tweet.text);
                if should_keep(&tweet, &cleaned, min_length, exclude_replies, only_replies)
                    && seen_cleaned.insert(cleaned.clone())
                {
                    filtered.push(FilteredTweet {
                        source: tweet.source,
                        id: tweet.id,
                        text: cleaned,
                        is_reply: tweet.is_reply,
                    });
                    if limit.is_some_and(|limit| filtered.len() >= limit) {
                        eprintln!(
                            "Loaded {} records from {} (kept {}, stopped at limit)",
                            raw_count - before_raw,
                            source_name,
                            filtered.len() - before_kept
                        );
                        return Ok((filtered, raw_count));
                    }
                }
            }
        }

        eprintln!(
            "Loaded {} records from {} (kept {})",
            raw_count - before_raw,
            source_name,
            filtered.len() - before_kept
        );
    }

    Ok((filtered, raw_count))
}

fn load_filtered_dms(
    path: &Path,
    owner_id: &str,
    min_length: usize,
    limit: Option<usize>,
) -> anyhow::Result<(Vec<FilteredTweet>, usize)> {
    let files = dm_files(path);
    if files.is_empty() {
        anyhow::bail!(
            "Could not find direct-messages.js or direct-messages-group.js under {}",
            path.display()
        );
    }

    let mut raw_count = 0;
    let mut seen_cleaned = HashSet::new();
    let mut filtered = Vec::new();

    for file in files {
        let data = parse_archive_json(&file)?;
        let source_name = file.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let before_raw = raw_count;
        let before_kept = filtered.len();

        for value in data {
            let Some(messages) = value
                .get("dmConversation")
                .and_then(|conversation| conversation.get("messages"))
                .and_then(Value::as_array)
            else {
                continue;
            };

            for message in messages {
                let Some(message_create) = message.get("messageCreate") else {
                    continue;
                };
                let Some(text) = message_create.get("text").and_then(Value::as_str) else {
                    continue;
                };
                let Some(sender_id) = message_create.get("senderId").and_then(Value::as_str) else {
                    continue;
                };

                raw_count += 1;
                if sender_id != owner_id {
                    continue;
                }

                let cleaned = clean_text(text);
                let dm = TweetText {
                    source: "direct-messages.js",
                    id: message_create
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    text: text.to_string(),
                    is_retweet: false,
                    is_reply: false,
                };

                if should_keep(&dm, &cleaned, min_length, false, false)
                    && seen_cleaned.insert(cleaned.clone())
                {
                    filtered.push(FilteredTweet {
                        source: source_name_to_static(source_name),
                        id: dm.id,
                        text: cleaned,
                        is_reply: false,
                    });
                    if limit.is_some_and(|limit| filtered.len() >= limit) {
                        eprintln!(
                            "Loaded {} DM records from {} (kept {}, stopped at limit)",
                            raw_count - before_raw,
                            source_name,
                            filtered.len() - before_kept
                        );
                        return Ok((filtered, raw_count));
                    }
                }
            }
        }

        eprintln!(
            "Loaded {} DM records from {} (kept {})",
            raw_count - before_raw,
            source_name,
            filtered.len() - before_kept
        );
    }

    Ok((filtered, raw_count))
}

fn source_name_to_static(source_name: &str) -> &'static str {
    match source_name {
        "direct-messages-group.js" => "direct-messages-group.js",
        "direct-messages.js" => "direct-messages.js",
        "note-tweet.js" => "note-tweet.js",
        _ => "tweets.js",
    }
}

fn clean_text(text: &str) -> String {
    let url_re = Regex::new(r"https?://\S+").unwrap();
    let leading_mentions_re = Regex::new(r"^((?:@\w+\s*)+)").unwrap();
    let ws_re = Regex::new(r"\s+").unwrap();
    let decoded = html_escape::decode_html_entities(text).to_string();
    let without_urls = url_re.replace_all(&decoded, "");
    let without_leading_mentions = leading_mentions_re.replace(without_urls.trim(), "");
    ws_re
        .replace_all(without_leading_mentions.trim(), " ")
        .trim()
        .to_string()
}

fn should_keep(
    tweet: &TweetText,
    cleaned: &str,
    min_length: usize,
    exclude_replies: bool,
    only_replies: bool,
) -> bool {
    if tweet.is_retweet {
        return false;
    }
    if exclude_replies && tweet.is_reply {
        return false;
    }
    if only_replies && !tweet.is_reply {
        return false;
    }

    if cleaned.len() < min_length {
        return false;
    }
    if contains_large_encoded_blob(cleaned) {
        return false;
    }

    // Drop if only hashtags/mentions remain
    let meaningful_words: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|w| !w.starts_with('#') && !w.starts_with('@'))
        .collect();

    meaningful_words
        .iter()
        .any(|word| word.chars().any(|c| c.is_alphanumeric()))
}

fn contains_large_encoded_blob(text: &str) -> bool {
    text.split_whitespace().any(|word| {
        let len = word.chars().count();
        len >= 120
            && word
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '=' | '_' | '-'))
    })
}

// ── Checkpoint loading ─────────────────────────────────────────────────────────

async fn load_checkpoint(output: &PathBuf) -> anyhow::Result<HashSet<String>> {
    let mut seen = HashSet::new();
    if !output.exists() {
        return Ok(seen);
    }

    let file = File::open(output).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        if let Ok(record) = serde_json::from_str::<AlpacaRecord>(&line) {
            seen.insert(record.output);
        }
    }

    eprintln!("Resuming — found {} existing records", seen.len());
    Ok(seen)
}

// ── Ollama instruction generation ──────────────────────────────────────────────

const SYSTEM_PROMPT: &str = r#"You generate training data for an LLM fine-tune.
Given a short text written by a specific person, write a SHORT, natural instruction
that someone might give to produce that text. The instruction should be generic
enough to be reusable, but specific enough to be meaningful.

Rules:
- Return ONLY a JSON object: {"instruction": "..."}
- No explanation, no markdown, no extra text
- Instruction should be 5-15 words
- Do not reference Twitter, DMs, messages, or social media in the instruction
- Examples:
  Text: "The best code is the code you never have to write"
  {"instruction": "Share a thought about writing clean, minimal code"}

  Text: "Austin traffic at 5pm is a special kind of hell"
  {"instruction": "Complain humorously about rush hour traffic"}"#;

const REPLY_GATE_PROMPT: &str = r#"You decide whether a reply can become useful LLM fine-tune data.
Given only the cleaned reply text, decide if someone could write a clear,
standalone instruction that would naturally produce this text.

Return ONLY a JSON object: {"can_generate": true} or {"can_generate": false}

Return false when the reply is mostly:
- dependent on missing conversation context
- just agreement, disagreement, laughter, emoji, or a reaction
- a short answer to an unknown question
- addressed to a specific person in a way that cannot be generalized

Return true when the reply expresses a complete opinion, explanation, joke,
technical answer, or other standalone thought."#;

enum GenerationOutcome {
    Generated(AlpacaRecord),
    Skipped(String),
}

fn ollama_content_from_body(body: &str) -> anyhow::Result<String> {
    let data: OllamaResponse = serde_json::from_str(body).map_err(|e| {
        anyhow::anyhow!("Could not parse Ollama response as chat JSON: {e}; body: {body}")
    })?;
    let content = data.message.content.trim().to_string();
    if content.is_empty() {
        let reason = data.done_reason.as_deref().unwrap_or("unknown");
        let thinking_preview = data
            .message
            .thinking
            .as_deref()
            .unwrap_or("")
            .chars()
            .take(240)
            .collect::<String>();
        anyhow::bail!(
            "Ollama returned empty message content (done_reason: {reason}). Thinking preview: {thinking_preview}"
        );
    }
    Ok(content)
}

fn openai_content_from_body(body: &str) -> anyhow::Result<String> {
    let data: OpenAiChatResponse = serde_json::from_str(body).map_err(|e| {
        anyhow::anyhow!("Could not parse OpenAI response as chat JSON: {e}; body: {body}")
    })?;
    let content = data
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .unwrap_or("")
        .trim()
        .to_string();
    if content.is_empty() {
        anyhow::bail!("OpenAI backend returned empty message content; body: {body}");
    }
    Ok(content)
}

async fn ollama_chat(client: &Client, payload: Value, base_url: &str) -> anyhow::Result<String> {
    let resp = client
        .post(format!("{}/api/chat", base_url.trim_end_matches('/')))
        .json(&payload)
        .send()
        .await?;

    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("Ollama returned HTTP {}: {}", status, body);
    }

    ollama_content_from_body(&body)
}

async fn openai_chat(
    client: &Client,
    payload: Value,
    base_url: &str,
    api_key: &str,
) -> anyhow::Result<String> {
    let resp = client
        .post(format!(
            "{}/v1/chat/completions",
            base_url.trim_end_matches('/')
        ))
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await?;

    let status = resp.status();
    let body = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("OpenAI backend returned HTTP {}: {}", status, body);
    }

    openai_content_from_body(&body)
}

async fn chat_json(
    client: &Client,
    backend: &BackendConfig,
    system_prompt: &str,
    user_prompt: String,
    temperature: f64,
    max_tokens: u64,
) -> anyhow::Result<String> {
    match backend.kind {
        BackendKind::Ollama => {
            let payload = serde_json::json!({
                "model": backend.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": false,
                "format": "json",
                "think": false,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            });
            ollama_chat(client, payload, &backend.base_url).await
        }
        BackendKind::Openai => {
            let Some(api_key) = backend.api_key.as_deref() else {
                anyhow::bail!("OpenAI backend requires an API key");
            };
            let payload = serde_json::json!({
                "model": backend.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"}
            });
            openai_chat(client, payload, &backend.base_url, api_key).await
        }
    }
}

async fn reply_can_generate_instruction(
    client: &Client,
    tweet: &str,
    backend: &BackendConfig,
) -> anyhow::Result<bool> {
    let content = chat_json(
        client,
        backend,
        REPLY_GATE_PROMPT,
        format!("Reply: \"{}\"", tweet),
        0.0,
        80,
    )
    .await?;
    let parsed: Value = serde_json::from_str(&content).map_err(|e| {
        anyhow::anyhow!("Reply gate did not return valid JSON: {e}; content: {content}")
    })?;

    parsed
        .get("can_generate")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Reply gate JSON did not include boolean field `can_generate`: {content}"
            )
        })
}

async fn generate_instruction(
    client: &Client,
    tweet: &str,
    backend: &BackendConfig,
) -> anyhow::Result<AlpacaRecord> {
    let mut content = chat_json(
        client,
        backend,
        SYSTEM_PROMPT,
        format!("Text: \"{}\"", tweet),
        0.1,
        200,
    )
    .await?;

    // Strip <think> blocks (reasoning models)
    let think_re = Regex::new(r"(?s)<think>.*?</think>").unwrap();
    content = think_re.replace_all(&content, "").trim().to_string();

    // Strip markdown fences
    content = content
        .replace("```json", "")
        .replace("```", "")
        .trim()
        .to_string();

    let parsed: Value = serde_json::from_str(&content).map_err(|e| {
        anyhow::anyhow!("Model did not return valid instruction JSON: {e}; content: {content}")
    })?;
    let instruction = parsed
        .get("instruction")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            anyhow::anyhow!("Model JSON did not include string field `instruction`: {content}")
        })?
        .trim()
        .to_string();

    if instruction.is_empty() {
        anyhow::bail!("Model returned an empty instruction");
    }

    Ok(AlpacaRecord {
        instruction,
        input: String::new(),
        output: tweet.to_string(),
    })
}

async fn process_tweet(
    client: &Client,
    tweet: &FilteredTweet,
    backend: &BackendConfig,
) -> anyhow::Result<GenerationOutcome> {
    if tweet.is_reply && !reply_can_generate_instruction(client, &tweet.text, backend).await? {
        return Ok(GenerationOutcome::Skipped(
            "reply too context-dependent for a useful instruction".to_string(),
        ));
    }

    let record = generate_instruction(client, &tweet.text, backend).await?;
    Ok(GenerationOutcome::Generated(record))
}

fn preview_records(label: &str, records: &[FilteredTweet]) {
    eprintln!("\n{} preview:", label);
    for (idx, tweet) in records.iter().take(25).enumerate() {
        let id = tweet.id.as_deref().unwrap_or("unknown");
        let kind = if tweet.is_reply { "reply" } else { "post" };
        eprintln!(
            "\n{}. [{}:{}:{}] {}",
            idx + 1,
            tweet.source,
            id,
            kind,
            tweet.text
        );
    }
    if records.len() > 25 {
        eprintln!("\n... {} more kept records not shown", records.len() - 25);
    }
}

async fn generate_dataset(
    label: &str,
    records: Vec<FilteredTweet>,
    output: &PathBuf,
    client: Arc<Client>,
    backend: Arc<BackendConfig>,
    workers: usize,
) -> anyhow::Result<()> {
    let seen = load_checkpoint(output).await?;
    let to_process: Vec<FilteredTweet> = records
        .into_iter()
        .filter(|tweet| !seen.contains(tweet.text.as_str()))
        .collect();

    eprintln!(
        "{} to process: {}  (already done: {})",
        label,
        to_process.len(),
        seen.len()
    );

    if to_process.is_empty() {
        eprintln!("{} dataset is complete.", label);
        return Ok(());
    }

    let pb = ProgressBar::new(to_process.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) | {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message("F:0 S:0");

    let mut out_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output)
        .await?;

    let semaphore = Arc::new(Semaphore::new(workers));
    let mut success_count = 0u64;
    let mut fail_count = 0u64;
    let mut skip_count = 0u64;
    let mut shown_failures = 0u64;
    let mut shown_skips = 0u64;

    let results = stream::iter(to_process)
        .map(|tweet| {
            let client = Arc::clone(&client);
            let backend = Arc::clone(&backend);
            let sem = Arc::clone(&semaphore);
            async move {
                let _permit = sem.acquire().await.unwrap();
                let result = process_tweet(&client, &tweet, &backend).await;
                (tweet, result)
            }
        })
        .buffer_unordered(workers);

    tokio::pin!(results);

    while let Some((tweet, record)) = results.next().await {
        match record {
            Ok(GenerationOutcome::Generated(r)) => {
                let line = serde_json::to_string(&r)? + "\n";
                out_file.write_all(line.as_bytes()).await?;
                success_count += 1;
            }
            Ok(GenerationOutcome::Skipped(reason)) => {
                skip_count += 1;
                if shown_skips < 5 {
                    pb.println(format!("Skipped: {} | {}", reason, tweet.text));
                    shown_skips += 1;
                }
            }
            Err(err) => {
                fail_count += 1;
                if shown_failures < 5 {
                    pb.println(format!("Generation failed: {err:#}"));
                    shown_failures += 1;
                }
            }
        }
        pb.set_message(format!("F:{} S:{}", fail_count, skip_count));
        pb.inc(1);

        if (success_count + fail_count + skip_count).is_multiple_of(100) {
            out_file.flush().await?;
        }
    }

    out_file.flush().await?;
    pb.finish_with_message(format!(
        "done — failed: {} skipped: {}",
        fail_count, skip_count
    ));

    let total = seen.len() as u64 + success_count;
    eprintln!(
        "\n{} done. Total records: {}  |  New: {}  |  Skipped: {}  |  Failed: {}",
        label, total, success_count, skip_count, fail_count
    );
    eprintln!("Output: {}", output.display());

    Ok(())
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let base_url = match args.backend {
        BackendKind::Ollama => args.ollama_url.clone(),
        BackendKind::Openai => args.openai_base_url.clone(),
    };
    let workers = args.workers.unwrap_or(match args.backend {
        BackendKind::Ollama => 1,
        BackendKind::Openai => 4,
    });

    eprintln!(
        "Using backend: {} | model: {} | base URL: {} | workers: {} | timeout: {}s | output: {} | dms output: {}",
        args.backend,
        args.model,
        base_url,
        workers,
        args.timeout_secs,
        args.output.display(),
        args.dms_output.display()
    );

    let mut tweet_records = Vec::new();
    if !args.dms_only {
        let (filtered, raw_count) = load_filtered_tweets(
            &args.archive,
            args.min_length,
            args.exclude_replies,
            args.only_replies,
            args.limit,
        )?;

        eprintln!(
            "Kept {} tweets after filtering and dedupe (from {} raw records)",
            filtered.len(),
            raw_count
        );
        tweet_records = filtered;
    }

    let mut dm_records = Vec::new();
    if args.include_dms || args.dms_only {
        let owner_id = load_owner_id(&args.archive, args.owner_id.as_deref())?;
        eprintln!("Using owner account id for DMs: {}", owner_id);
        let (filtered, raw_count) =
            load_filtered_dms(&args.archive, &owner_id, args.min_length, args.limit)?;

        eprintln!(
            "Kept {} outbound DMs after filtering and dedupe (from {} raw DM records)",
            filtered.len(),
            raw_count
        );
        dm_records = filtered;
    }

    if args.dry_run {
        if !tweet_records.is_empty() {
            preview_records("Tweets", &tweet_records);
        }
        if !dm_records.is_empty() {
            preview_records("DMs", &dm_records);
        }
        return Ok(());
    }

    let api_key = match args.backend {
        BackendKind::Ollama => None,
        BackendKind::Openai => Some(args.api_key.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "Missing --api-key or OPENAI_API_KEY for backend `{}`",
                args.backend
            )
        })?),
    };
    let backend = BackendConfig {
        kind: args.backend,
        model: args.model.clone(),
        base_url,
        api_key,
    };

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout_secs))
        .build()?;

    if matches!(backend.kind, BackendKind::Ollama) {
        client
            .get(format!(
                "{}/api/tags",
                backend.base_url.trim_end_matches('/')
            ))
            .send()
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Cannot reach Ollama at {} — is it running?",
                    backend.base_url
                )
            })?;
    }

    let client = Arc::new(client);
    let backend = Arc::new(backend);

    if !tweet_records.is_empty() {
        generate_dataset(
            "Tweets",
            tweet_records,
            &args.output,
            Arc::clone(&client),
            Arc::clone(&backend),
            workers,
        )
        .await?;
    }

    if !dm_records.is_empty() {
        generate_dataset(
            "DMs",
            dm_records,
            &args.dms_output,
            client,
            backend,
            workers,
        )
        .await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleans_urls_entities_and_leading_mentions() {
        let cleaned = clean_text("@alice @bob This &gt; that https://t.co/example\n\nok");
        assert_eq!(cleaned, "This > that ok");
    }

    #[test]
    fn keeps_substantive_replies_when_allowed() {
        let tweet = TweetText {
            source: "tweets.js",
            id: Some("1".to_string()),
            text: "@alice Tokens are expensive".to_string(),
            is_retweet: false,
            is_reply: true,
        };
        let cleaned = clean_text(&tweet.text);

        assert!(should_keep(&tweet, &cleaned, 10, false, false));
        assert!(!should_keep(&tweet, &cleaned, 10, true, false));
        assert!(should_keep(&tweet, &cleaned, 10, false, true));
    }

    #[test]
    fn only_replies_drops_non_replies() {
        let tweet = TweetText {
            source: "tweets.js",
            id: Some("1".to_string()),
            text: "This is a normal standalone post".to_string(),
            is_retweet: false,
            is_reply: false,
        };
        let cleaned = clean_text(&tweet.text);

        assert!(should_keep(&tweet, &cleaned, 10, false, false));
        assert!(!should_keep(&tweet, &cleaned, 10, false, true));
    }

    #[test]
    fn drops_retweets() {
        let tweet = TweetText {
            source: "tweets.js",
            id: Some("1".to_string()),
            text: "RT @alice useful text that should not matter".to_string(),
            is_retweet: true,
            is_reply: false,
        };
        let cleaned = clean_text(&tweet.text);

        assert!(!should_keep(&tweet, &cleaned, 10, false, false));
    }

    #[test]
    fn drops_large_encoded_blobs() {
        let blob = "a".repeat(160);
        let tweet = TweetText {
            source: "direct-messages.js",
            id: Some("1".to_string()),
            text: blob.clone(),
            is_retweet: false,
            is_reply: false,
        };

        assert!(contains_large_encoded_blob(&blob));
        assert!(!should_keep(&tweet, &blob, 10, false, false));
    }

    #[test]
    fn archive_data_dir_handles_archive_root_and_data_dir() {
        assert_eq!(
            archive_data_dir(Path::new("/tmp/archive")),
            PathBuf::from("/tmp/archive/data")
        );
        assert_eq!(
            archive_data_dir(Path::new("/tmp/archive/data")),
            PathBuf::from("/tmp/archive/data")
        );
    }

    #[test]
    fn source_name_to_static_handles_dm_sources() {
        assert_eq!(
            source_name_to_static("direct-messages-group.js"),
            "direct-messages-group.js"
        );
        assert_eq!(
            source_name_to_static("direct-messages.js"),
            "direct-messages.js"
        );
    }

    #[test]
    fn backend_kind_displays_cli_values() {
        assert_eq!(BackendKind::Ollama.to_string(), "ollama");
        assert_eq!(BackendKind::Openai.to_string(), "openai");
    }

    #[test]
    fn parses_ollama_content() {
        let body = r#"{"message":{"content":"{\"instruction\":\"Do a thing\"}","thinking":null},"done_reason":"stop"}"#;
        assert_eq!(
            ollama_content_from_body(body).unwrap(),
            r#"{"instruction":"Do a thing"}"#
        );
    }

    #[test]
    fn parses_openai_compatible_content() {
        let body = r#"{"choices":[{"message":{"content":"{\"instruction\":\"Do a thing\"}"}}]}"#;
        assert_eq!(
            openai_content_from_body(body).unwrap(),
            r#"{"instruction":"Do a thing"}"#
        );
    }
}
