## infinex

>"a scalabale  inference engine that scales to infinity and beyond"


building a inference engine from scratch from first principles. 

built in Go to be fast and highly distributed from the ground up and no high abstraction layers that obscure what is actually happening.

a broader goal is to build a minimal viable grade inference server with PagedAttention, request batching, weight quantization and an OpenAI compatible HTTP API.


### load the model via GGUF files 

User should be able to download any GGUF file from HuggingFace and pass it directly to infinex.

Download the GGUF files using hf cli and place it under `data/`
Eg: for GPT-2 , you would work it like this 
```bash
hf download openai-community/gpt2 --local-dir data/gpt2
```

### Commands Supported (High Level)

1. one shot (simplest)
  - user will give the gguf files of he models and the prompt as a cli and the max tokens size
  and the engine will do inference and give the output.

2. interactive (chat loop)
  - open a tui loop with lipgloss as a TUI framework to continunously work (until unless max token reached)

