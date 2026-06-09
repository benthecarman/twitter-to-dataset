# Twitter Data Parser

Convert a personal Twitter/X archive into an Alpaca-format JSONL dataset.

The tool reads archive files such as `data/tweets.js` and `data/note-tweet.js`,
cleans and filters the text, then uses a local Ollama model to generate a short
instruction for each output record.

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

## Usage

Preview records without calling Ollama or writing output:

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

Replies are gated by an extra LLM check before instruction generation. Replies
that are too context-dependent are skipped instead of written to the dataset.

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
