import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import snapshot_download
import torch


def load_model(model_name_or_path, download=False, cache_dir=None):
    """Load a tokenizer and model from a local path or the Hugging Face Hub."""
    if download:
        local_dir = snapshot_download(model_name_or_path, cache_dir=cache_dir, resume_download=True)
        path = local_dir
    else:
        path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)

    return tokenizer, model


def chat_loop(tokenizer, model, device="cpu"):
    model.to(device)
    while True:
        user_input = input("User> ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, max_new_tokens=128, do_sample=True, top_p=0.95)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Print only the generated portion
        print("Bot>", response[len(user_input):].strip())


def main():

    parser = argparse.ArgumentParser(description="Chat with a local or Hugging Face model")
    parser.add_argument("--model", default="gpt2", help="Path or model ID (default: gpt2)")
    parser.add_argument("--download", action="store_true", help="Download the model from the Hugging Face Hub")
    parser.add_argument("--cache-dir", help="Directory for downloaded models")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model, download=args.download, cache_dir=args.cache_dir)

    chat_loop(tokenizer, model, device)


if __name__ == "__main__":
    main()

