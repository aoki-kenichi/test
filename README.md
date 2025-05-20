# Local LLM Chat Program

This repository provides a simple Python script that allows you to chat with a lightweight language model stored on your machine.

## Requirements

- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [torch](https://pytorch.org/)

Install dependencies with pip:

```bash
pip install transformers torch
```

## Usage

1. Download a compatible causal language model (e.g. GPT-Neo, Llama) and keep it on your local filesystem.
2. Run the script with the path to the downloaded model:

```bash
python chat_local_llm.py --model-path /path/to/your/model
```

Type `quit` or `exit` to end the chat session.

