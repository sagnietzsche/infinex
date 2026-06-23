## infinex

building a inference engine from scratch from first principles.

built in Go to be highly distributed from the ground up.

"supposed to be scale this inference engine to infinity and beyond"

something of a mini ollama


### GGUF

Download the GGUF files using hf cli and place it under `data/`
```bash
hf download openai-community/gpt2 --local-dir data/gpt2
```

### Commands Supported (High Level)

1. one shot (simplest)
  - user will give the gguf files of he models and the prompt as a cli and the max tokens size
  and the engine will do inference and give the output.

2. interactive (chat loop)
  - open a tui loop with lipgloss as a TUI framework to continunously work (until unless max token reached)

### ROADMAP

### Week 1 — "It runs"

**One deliverable:** `llm-go run --model gpt2.gguf --prompt "hello" --max-tokens 50` prints tokens to stdout.

That's it. Everything in week 1 is in service of that single command working.

#### What you build, in order:

**Day 1–2: Tensor + basic ops**
- `Tensor` struct with `[]float32` backing + shape
- `MatMul`, `Add`, `Mul`, `Transpose`
- `Softmax`, `RMSNorm` (or `LayerNorm` for GPT-2)
- Write unit tests against known values — this will save you days of debugging later

**Day 3: GGUF parser**
- Binary file reader that produces `map[string]*Tensor`
- Print all layer names + shapes to verify it loaded correctly
- No real logic here, just file I/O

**Day 4–5: GPT-2 forward pass**
- Token embedding lookup
- Single `TransformerBlock` (attention + MLP + norms)
- Stack N blocks
- Final LM head projection → logits

**Day 6: Greedy sampler + tokenizer**
- Wire in `tiktoken-go`
- Argmax over logits → next token ID → decode to string
- No KV cache yet — recompute everything each step (slow but correct)

**Day 7: Glue it together**
- `cobra` CLI skeleton
- `generate()` loop: tokens in → stream strings out via channel
- Verify output makes sense (not garbage)

#### What you explicitly do NOT build in week 1:
- KV cache
- Any sampler other than greedy
- Chat loop
- HTTP server
- Anything beyond GPT-2


### Week 2 — "It's usable"

**Two deliverables:**
1. `llm-go run` is now fast and has real sampling
2. `llm-go chat` drops you into an interactive REPL

#### What you build, in order:

**Day 8–9: KV cache**
- Pre-allocate K/V tensors per layer
- On each decode step, append new K/V, attend over full cache
- This is the single biggest speed jump you'll get — generation goes from O(n²) per token to O(n)

**Day 10: Samplers**
- Temperature scaling
- Top-p (nucleus sampling)
- Expose via `--temperature` and `--top-p` flags

**Day 11: Chat loop**
- `llm-go chat --model gpt2.gguf`
- readline REPL, append user turns to context, stream assistant response
- Keep a `[]int` conversation history, truncate when it exceeds context window

**Day 12–13: LLaMA/Mistral support**
- RoPE positional embeddings (replaces GPT-2's learned embeddings)
- SwiGLU activation (replaces GELU in MLP)
- RMSNorm (replaces LayerNorm)
- Grouped Query Attention (GQA) — Mistral uses this
- Test with TinyLlama 1.1B

**Day 14: Polish**
- `--system` prompt flag for chat mode
- Token/sec throughput printed after generation
- Graceful context window overflow handling
- Clean up error messages

### The arc in one sentence each

**Week 1:** Prove the math works. Slow, greedy, GPT-2 only — but correct.

**Week 2:** Make it fast, usable, and model-agnostic. Something you'd actually want to use.

By end of week 2 you have something real: a CLI inference engine that loads open-weight GGUF models and runs interactive chat. That's genuinely a useful, shippable open source tool.

