import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
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
    parser = argparse.ArgumentParser(description="Chat with a local lightweight LLM")
    parser.add_argument("--model-path", required=True, help="Path to the local model directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    chat_loop(tokenizer, model, device)


if __name__ == "__main__":
    main()

