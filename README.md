# Twitter Data Parser

Convert a personal Twitter/X archive into an Alpaca-format JSONL dataset.

The tool reads archive files such as `data/tweets.js` and `data/note-tweet.js`,
cleans and filters the text, then uses the selected model backend to generate
a short instruction for each output record. Ollama is the default backend.

## Data Quality Strategies

The archive contains a lot of text that does not make good instruction-tuning
data on its own. The tool uses a few filters before writing records:

- Text cleanup removes URLs, leading mentions, HTML entities, and excess
  whitespace.
- Basic filtering skips retweets, very short posts, mention-only posts,
  hashtag-only posts, and likely encoded blobs.
- Deduplication skips repeated cleaned text.
- Before instruction generation, public tweets are checked by the selected
  model for standalone usefulness. Posts that depend on missing thread, link,
  image, event, or quoted-post context are skipped.
- Replies can be processed separately with `--only-replies` or excluded with
  `--exclude-replies`. When replies are included, the tool first asks the model
  whether the reply is standalone enough to become a useful instruction/output
  pair. Context-dependent replies are skipped.
- DMs are written to a separate dataset with `--dms-only` or `--include-dms`.
  Only outbound DMs are considered. Before instruction generation, the tool
  skips messages that look too context-dependent, too private, or likely contain
  private contact information such as addresses, phone numbers, or email
  addresses.

## Getting Your Twitter/X Archive

In X/Twitter:

1. Open **Settings and privacy**.
2. Go to **Your account**.
3. Select **Download an archive of your data**.
4. Complete the verification flow and request the archive.
5. When X emails or notifies you that it is ready, download the `.zip`.
6. Extract it locally.

This tool expects either the extracted archive directory or its `data/`
directory. For example:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --dry-run \
  --limit 25
```

The extracted archive should contain files like:

```text
data/tweets.js
data/note-tweet.js
data/direct-messages.js
data/account.js
```

## Installation

Install from a GitHub release:

```bash
curl -L https://github.com/benthecarman/twitter-to-dataset/releases/latest/download/twitter-to-dataset-v0.1.0-x86_64-unknown-linux-gnu.tar.gz |
  tar -xz
sudo mv twitter-to-dataset /usr/local/bin/
```

Replace the archive name with the macOS artifact if needed:

```text
twitter-to-dataset-v0.1.0-x86_64-apple-darwin.tar.gz
twitter-to-dataset-v0.1.0-aarch64-apple-darwin.tar.gz
```

Install with Cargo after the crate is published:

```bash
cargo install twitter-to-dataset
```

Install with Cargo from GitHub:

```bash
cargo install --git https://github.com/benthecarman/twitter-to-dataset
```

Or run from a local checkout:

```bash
cargo run -- --help
```

## Shell Completions

Generate completions with:

```bash
twitter-to-dataset --generate-completions <shell>
```

Supported shells are `bash`, `elvish`, `fish`, `powershell`, and `zsh`.

Examples:

```bash
# bash
twitter-to-dataset --generate-completions bash > /etc/bash_completion.d/twitter-to-dataset

# zsh
twitter-to-dataset --generate-completions zsh > "${fpath[1]}/_twitter-to-dataset"

# fish
twitter-to-dataset --generate-completions fish > ~/.config/fish/completions/twitter-to-dataset.fish

# PowerShell
twitter-to-dataset --generate-completions powershell > twitter-to-dataset.ps1
```

## Usage

Preview records without calling a model backend or writing output:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --dry-run \
  --limit 25
```

Generate a small test dataset:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --limit 25 \
  --output test-dataset.jsonl \
  --model qwen3:14b
```

Use the OpenAI backend instead of Ollama:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --limit 25 \
  --output test-dataset.jsonl \
  --backend openai \
  --api-key sk-... \
  --model gpt-4.1-mini
```

Ollama is the default because it keeps archive data local. The OpenAI backend
is explicit opt-in. The API key can be passed with `--api-key` or set with
`OPENAI_API_KEY`.

For another OpenAI-compatible server, set its base URL:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --limit 25 \
  --backend openai \
  --openai-base-url http://localhost:1234 \
  --api-key local-key \
  --model local-model
```

Use custom data quality prompts:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --tweet-prompt prompts/tweets.txt \
  --reply-prompt prompts/replies.txt \
  --dm-prompt prompts/dms.txt \
  --output dataset.jsonl \
  --model qwen3:14b
```

Each prompt file replaces the built-in quality gate prompt for that record
type. Custom prompts must still instruct the model to return only
`{"can_generate": true}` or `{"can_generate": false}`.

Generate standalone posts first:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --exclude-replies \
  --output dataset-posts.jsonl \
  --model qwen3:14b
```

Generate replies afterward:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --only-replies \
  --output dataset-replies.jsonl \
  --model qwen3:14b
```

Generate a separate outbound DM dataset:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --dms-only \
  --dms-output dms-dataset.jsonl \
  --model qwen3:14b
```

Generate tweets and outbound DMs in one run, still writing separate files:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --include-dms \
  --output dataset.jsonl \
  --dms-output dms-dataset.jsonl \
  --model qwen3:14b
```

DM parsing uses `data/account.js` to detect your account id and only includes
messages sent by you. Use `--owner-id` if the account file is unavailable.

For slow local models:

```bash
cargo run -- \
  --archive ~/Downloads/twitter-2026-06-03-archive \
  --limit 5 \
  --output test-dataset.jsonl \
  --model nemotron-3-super:latest \
  --workers 1 \
  --timeout-secs 900
```

`--workers` is optional. When omitted, it defaults to `1` for Ollama and `4`
for the OpenAI backend.

## Output Format

Each generated line is an Alpaca-style JSON object:

```json
{"instruction":"Explain a privacy concern about a payment system","input":"","output":"..."}
```

## Notes

- `--dry-run` only parses, cleans, filters, dedupes, and prints examples.
- `--limit` stops early after enough kept records are found.
- Existing output files are used as checkpoints, so reruns skip already-written
  outputs.
- Generated `.jsonl` files are ignored by git.
- Deleted tweets and deleted note tweets are intentionally not included.

## Releases

Pushing a version tag creates a GitHub release with Linux and macOS binaries:

```bash
git tag v0.1.0
git push origin v0.1.0
```
