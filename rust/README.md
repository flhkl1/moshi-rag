# MoshiRAG - Rust

![License](https://img.shields.io/crates/l/moshi.svg)

See the [top-level README.md](../README.md) for more information.

This provides the Rust backend (including MoshiRAG with ARC-Encoder, Mimi, and the STT model) and client implementation.

## Requirements

You will need a recent version of the [Rust toolchain](https://rustup.rs/).
To compile GPU support, you will also need the [CUDA](https://developer.nvidia.com/cuda-toolkit) properly installed for your GPU, in particular with `nvcc`.


## Rust server for MoshiRAG

If you don't have ssl certificates yet, generate a `key.pem` and `cert.pem` file
using the following command.
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

### MoshiRAG configuration

Use a config where the required paths are set. Either copy `moshi-backend/config.json` and edit it, or add these keys to your existing config:
   - `lm_model_file` - path to the main MoshiRAG LM safetensors
   - `text_tokenizer_file` - path to the text tokenizer of the main MoshiRAG LM
   - `mimi_model_file` - path to the mimi safetensor checkpoint
   - `stt_lm_model_file` — path to the STT LM safetensors
   - `stt_text_tokenizer_file` — path to the STT text tokenizer (e.g. `.model`)
   - `stt_mimi_model_file` — path to the STT Mimi checkpoint (can be the same as `mimi_model_file` if you use one Mimi for both)
   - `arc_encoder_tokenizer_path` - path to the text tokenizer of ARC-Encoder
   - `arc_encoder_model_file` - path to the ARC-Encoder safetensors checkpoint

MoshiRAG calls an OpenAI-compatible chat endpoint for retrieval. You can choose a local vLLM server or an online API.

- Local vLLM retrieval server:
  - Start vLLM with your retrieval model exposed at `/v1/chat/completions`.
    ```bash
    vllm serve google/gemma-3-27b-it --host 0.0.0.0 --port 8002
    ```
  - Set `LLM_BASE_URL=http://127.0.0.1:8002/v1`
  - Set `LLM_MODEL_NAME=<your-vllm-model-name>`
  - Set `LLM_API_KEY=""`
- Online API (OpenAI-compatible provider; choose low-latency service to guarantee MoshiRAG response quality):
  - Set `LLM_BASE_URL=<provider-base-url>/v1`
  - Set `LLM_MODEL_NAME=<provider-model-name>`
  - Set `LLM_API_KEY=<your-api-key>`
- Multiple retrieval LLMs: set **`MOSHI_RETRIEVAL_LLMS_JSON`** to a JSON array of objects with `id`, `base_url`, `model`, optional `api_key`, and exactly one `"default": true` when you list two or more profiles. In Rust, this can only be set through environment variables. When set and non-empty, it replaces single-endpoint `LLM_BASE_URL`/`LLM_MODEL_NAME` retrieval selection.
  - Example:
    ```bash
    MOSHI_RETRIEVAL_LLMS_JSON='[{"id": "gpt-oss-20b", "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b", "prompt_style": "simplified"},  {"id": "gemma-3-27b-it", "base_url": "http://localhost:8002/v1", "model": "google/gemma-3-27b-it", "default": true, "prompt_style": "original"}]' LLM_API_KEY=...
    ```
  - In this example:
    - `gpt-oss-20b` uses the `simplified` bundled reference prompt.
    - `gemma-3-27b-it` is marked as the default fallback profile via `default: true`.
    - If a profile does not set `api_key`, Rust uses global `LLM_API_KEY`.
    - The prompt style of each individual profile may be `"original"` or `"simplified"` (default `simplified`).

### Start the server
Run the comment below to start the MoshiRAG server (MoshiRAG model with ARC encoder + Kyutai STT ASR + mimi).

```bash
# From the rust/ directory (use --features metal on macOS)
export MOSHI_RETRIEVAL_LLMS_JSON='[{"id": "gpt-oss-20b", "base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b", "prompt_style": "simplified"},  {"id": "gemma-3-27b-it", "base_url": "http://localhost:8002/v1", "model": "google/gemma-3-27b-it", "default": true, "prompt_style": "original"}]'
export LLM_API_KEY=...

cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
```

When using macOS, you can replace `--features cuda` with `--features metal`.

Once the server prints `standalone worker listening`, open the web UI at
[localhost:8998](https://localhost:8998) (HTTPS by default).

You will get some warnings about the site being unsafe. When using chrome you
can bypass it by selecting "Details" or "Advanced", then "Visit this unsafe
site" or "Proceed to localhost (unsafe)".

## Rust client

We recommend using the web UI as it provides some echo cancellation that helps
the overall model quality.

## License

The present code is provided under the Apache license.
