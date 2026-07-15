"""infinex: a from-scratch LLM inference server.

Phases:
  1. sampling    -- softmax / temperature / top-k / top-p primitives
  2. tokenizer   -- char-level tokenizer with special tokens
  3. model       -- tiny transformer + contiguous KV cache (kv_cache)
  4. paged_kv    -- PagedAttention-style block pool + block tables
  5. sequence    -- Sequence lifecycle + static batching
  6. scheduler   -- continuous batching engine (Orca-style)
  7. server      -- streaming serving API
  8. bench       -- load generator + latency/throughput metrics
"""
