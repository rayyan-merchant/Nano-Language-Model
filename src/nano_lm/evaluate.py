"""Same-byte conditional BPB with explicit token-window semantics."""
import math
import torch
from torch.nn import functional as F
from .bpe import encode_with_bytes


def byte_prefix(text, byte_limit):
    if byte_limit < 0:
        raise ValueError("Negative byte limit")
    raw = text.encode("utf-8")
    return raw[:byte_limit].decode("utf-8", errors="ignore")


def evaluation_tokens(tokenizer, text, conditioning_bytes=256, max_bytes=0):
    if max_bytes:
        text = byte_prefix(text, max_bytes)
    prefix = byte_prefix(text, conditioning_bytes)
    suffix = text[len(prefix):]
    if not prefix or not suffix:
        raise ValueError("Evaluation needs a nonempty conditioning prefix and suffix")
    before, _ = encode_with_bytes(tokenizer, prefix)
    after, lengths = encode_with_bytes(tokenizer, suffix)
    return before + after, len(before), lengths, len(prefix.encode()), len(text.encode())


def metrics(nll_sum, tokens, source_bytes):
    if tokens <= 0 or source_bytes <= 0 or not math.isfinite(nll_sum):
        raise ValueError("Invalid likelihood/coverage")
    loss = nll_sum / tokens
    return {"loss": loss, "perplexity": math.exp(loss),
            "bpb": nll_sum / (math.log(2) * source_bytes), "nll_sum": nll_sum,
            "scored_tokens": tokens, "scored_bytes": source_bytes}


@torch.no_grad()
def evaluate(model, tokenizer, text, conditioning_bytes=256, max_bytes=0,
             batch_size=32, mode="sliding"):
    if mode not in {"sliding", "blocks"} or batch_size < 1:
        raise ValueError("Invalid evaluation mode/batch size")
    ids, start, lengths, prefix_bytes, total_bytes = evaluation_tokens(
        tokenizer, text, conditioning_bytes, max_bytes)
    device = next(model.parameters()).device
    context = model.config.context_length
    was_training = model.training
    model.eval()
    nll, count = 0.0, 0
    try:
        if mode == "sliding":
            # Equal-length windows can batch; prefix-adjacent shorter windows
            # are separate. Each suffix token gets maximal available context.
            group, group_length = [], None
            def score(indices):
                nonlocal nll, count
                windows = [ids[max(0, i - context):i] for i in indices]
                x = torch.tensor(windows, device=device)
                targets = torch.tensor([ids[i] for i in indices], device=device)
                logits = model(x)[0][:, -1].float()
                nll += F.cross_entropy(logits, targets, reduction="sum").double().item()
                count += len(indices)
            for i in range(start, len(ids)):
                length = min(context, i)
                if group and (length != group_length or len(group) == batch_size):
                    score(group)
                    group = []
                group.append(i)
                group_length = length
            if group:
                score(group)
        else:
            # Efficient nonoverlapping target blocks, one prior token context
            # at the boundary, increasing context within each block.
            for i in range(start, len(ids), context):
                stop = min(i + context, len(ids))
                x = torch.tensor([ids[i - 1:stop - 1]], device=device)
                y = torch.tensor(ids[i:stop], device=device)
                nll += F.cross_entropy(model(x)[0][0].float(), y, reduction="sum").double().item()
                count += stop - i
    finally:
        model.train(was_training)
    result = metrics(nll, count, sum(lengths))
    result.update({"conditioning_bytes": prefix_bytes, "evaluated_text_bytes": total_bytes,
                   "mode": mode, "context_length": context})
    return result
