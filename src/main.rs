use clap::Parser;
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
    name = "twitter_to_dataset",
    about = "Convert Twitter archive to Alpaca fine-tune dataset"
)]
struct Args {
    /// Path to a Twitter archive directory, data directory, tweets.js, or note-tweet.js
    #[arg(long)]
    archive: PathBuf,

    /// Output JSONL file
    #[arg(long, default_value = "dataset.jsonl")]
    output: PathBuf,

    /// Ollama model to use for instruction generation
    #[arg(long, default_value = "qwen3:14b")]
    model: String,

    /// Ollama base URL
    #[arg(long, default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Number of concurrent Ollama requests
    #[arg(long, default_value_t = 4)]
    workers: usize,

    /// Timeout for each Ollama request in seconds
    #[arg(long, default_value_t = 300)]
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

    /// Print kept examples and exit without calling Ollama or writing output
    #[arg(long)]
    dry_run: bool,
}

// ── Types ──────────────────────────────────────────────────────────────────────

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

    // Drop if only hashtags/mentions remain
    let meaningful_words: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|w| !w.starts_with('#') && !w.starts_with('@'))
        .collect();

    meaningful_words
        .iter()
        .any(|word| word.chars().any(|c| c.is_alphanumeric()))
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
Given a tweet written by a specific person, write a SHORT, natural instruction
that someone might give to produce that tweet. The instruction should be generic
enough to be reusable, but specific enough to be meaningful.

Rules:
- Return ONLY a JSON object: {"instruction": "..."}
- No explanation, no markdown, no extra text
- Instruction should be 5-15 words
- Do not reference Twitter, tweets, or social media in the instruction
- Examples:
  Tweet: "The best code is the code you never have to write"
  {"instruction": "Share a thought about writing clean, minimal code"}

  Tweet: "Austin traffic at 5pm is a special kind of hell"
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

async fn ollama_chat(client: &Client, payload: Value, ollama_url: &str) -> anyhow::Result<String> {
    let resp = client
        .post(format!("{}/api/chat", ollama_url))
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

async fn reply_can_generate_instruction(
    client: &Client,
    tweet: &str,
    model: &str,
    ollama_url: &str,
) -> anyhow::Result<bool> {
    let payload = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": REPLY_GATE_PROMPT},
            {"role": "user", "content": format!("Reply: \"{}\"", tweet)},
        ],
        "stream": false,
        "format": "json",
        "think": false,
        "options": {"temperature": 0.0, "num_predict": 80}
    });

    let content = ollama_chat(client, payload, ollama_url).await?;
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
    model: &str,
    ollama_url: &str,
) -> anyhow::Result<AlpacaRecord> {
    let payload = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format!("Tweet: \"{}\"", tweet)},
        ],
        "stream": false,
        "format": "json",
        "think": false,
        "options": {"temperature": 0.1, "num_predict": 200}
    });

    let mut content = ollama_chat(client, payload, ollama_url).await?;

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
    model: &str,
    ollama_url: &str,
) -> anyhow::Result<GenerationOutcome> {
    if tweet.is_reply
        && !reply_can_generate_instruction(client, &tweet.text, model, ollama_url).await?
    {
        return Ok(GenerationOutcome::Skipped(
            "reply too context-dependent for a useful instruction".to_string(),
        ));
    }

    let record = generate_instruction(client, &tweet.text, model, ollama_url).await?;
    Ok(GenerationOutcome::Generated(record))
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    eprintln!(
        "Using model: {} | workers: {} | timeout: {}s | output: {}",
        args.model,
        args.workers,
        args.timeout_secs,
        args.output.display()
    );

    // Load, clean, filter, and dedupe tweets.
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

    if args.dry_run {
        for (idx, tweet) in filtered.iter().take(25).enumerate() {
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
        if filtered.len() > 25 {
            eprintln!("\n... {} more kept records not shown", filtered.len() - 25);
        }
        return Ok(());
    }

    // Load checkpoint
    let seen = load_checkpoint(&args.output).await?;
    let to_process: Vec<FilteredTweet> = filtered
        .into_iter()
        .filter(|tweet| !seen.contains(tweet.text.as_str()))
        .collect();

    eprintln!(
        "To process: {}  (already done: {})",
        to_process.len(),
        seen.len()
    );

    if to_process.is_empty() {
        eprintln!("Nothing to do — dataset is complete!");
        return Ok(());
    }

    // Verify Ollama is reachable
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(args.timeout_secs))
        .build()?;

    client
        .get(format!("{}/api/tags", args.ollama_url))
        .send()
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Cannot reach Ollama at {} — is it running?",
                args.ollama_url
            )
        })?;

    // Progress bar
    let pb = ProgressBar::new(to_process.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) | {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message("F:0 S:0");

    // Open output file for appending
    let mut out_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
        .await?;

    let semaphore = Arc::new(Semaphore::new(args.workers));
    let client = Arc::new(client);
    let model = Arc::new(args.model.clone());
    let ollama_url = Arc::new(args.ollama_url.clone());

    let mut success_count = 0u64;
    let mut fail_count = 0u64;
    let mut skip_count = 0u64;
    let mut shown_failures = 0u64;
    let mut shown_skips = 0u64;

    let results = stream::iter(to_process)
        .map(|tweet| {
            let client = Arc::clone(&client);
            let model = Arc::clone(&model);
            let ollama_url = Arc::clone(&ollama_url);
            let sem = Arc::clone(&semaphore);
            async move {
                let _permit = sem.acquire().await.unwrap();
                let result = process_tweet(&client, &tweet, &model, &ollama_url).await;
                (tweet, result)
            }
        })
        .buffer_unordered(args.workers);

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
                    pb.println(format!("Skipped reply: {} | {}", reason, tweet.text));
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

        // Flush every 100 records
        if (success_count + fail_count + skip_count) % 100 == 0 {
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
        "\n✅  Done! Total records: {}  |  New: {}  |  Skipped: {}  |  Failed: {}",
        total, success_count, skip_count, fail_count
    );
    eprintln!("Output: {}", args.output.display());

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
}
