# Twitter Data Parser

Convert a personal Twitter/X archive into an Alpaca-format JSONL dataset.

The tool reads archive files such as `data/tweets.js` and `data/note-tweet.js`,
cleans and filters the text, then uses a local Ollama model to generate a short
instruction for each output record.

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
