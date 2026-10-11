"""Autoregressive generation and own-checkpoint command line inference."""
import argparse
from pathlib import Path
import torch
from .checkpoint import load_for_inference


@torch.no_grad()
def generate(model, ids, max_new_tokens=80, temperature=1.0, greedy=False, seed=42):
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError("Nonnegative length and positive temperature required")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=ids.device).manual_seed(seed)
    try:
        for _ in range(max_new_tokens):
            logits, _, _ = model(ids[:, -model.config.context_length:])
            logits = logits[:, -1] / temperature
            next_id = (logits.argmax(-1, keepdim=True) if greedy else
                       torch.multinomial(logits.softmax(-1), 1, generator=generator))
            ids = torch.cat((ids, next_id), dim=1)
    finally:
        model.train(was_training)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    model, tokenizer = load_for_inference(args.checkpoint)
    if not args.prompt:
        parser.error("Prompt cannot be empty")
    ids = torch.tensor([tokenizer.encode(args.prompt)])
    output = generate(model, ids, args.max_new_tokens, args.temperature, args.greedy, args.seed)
    print(tokenizer.decode(output[0].tolist()))
    print(f"parameters={model.parameter_count()} context={model.config.context_length} "
          f"temperature={args.temperature} seed={args.seed} checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
