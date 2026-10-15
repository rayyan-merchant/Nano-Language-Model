"""Canonical whole-sequence decoding and per-call seeded generation."""
import secrets
import time
import torch
from nano_lm.generate import generate
from .schemas import GenerationResult


def run(registry, request, request_id, seed=None):
    start = time.perf_counter()
    model = registry.load(request.model_id)
    tokenizer = registry.tokenizer
    encoded = tokenizer.encode(request.prompt)
    if not encoded:
        raise ValueError('Prompt must encode to at least one token')
    actual_seed = seed if seed is not None else request.seed if request.seed is not None else secrets.randbelow(2147483648)
    ids = torch.tensor([encoded], device=registry.device)
    output = generate(model, ids, request.max_new_tokens, request.temperature, request.greedy, actual_seed)
    raw = output[0].cpu().tolist()
    # Decode complete sequences, never individual byte tokens. The prompt ends on a
    # valid UTF-8 boundary, so suffix decoding equals canonical full continuation.
    full = tokenizer.decode(raw)
    continuation = tokenizer.decode(raw[len(encoded):])
    return GenerationResult(
        model_id=request.model_id, model_name=registry.entries[request.model_id]['name'],
        checkpoint_step=registry.entries[request.model_id]['checkpoint_step'],
        prompt=request.prompt, continuation=continuation, full_output=full,
        generated_tokens=len(raw)-len(encoded), output_characters=len(continuation),
        prompt_tokens=len(encoded), effective_context=min(len(encoded), model.config.context_length),
        context_cropped=len(encoded)>model.config.context_length,
        settings={'max_new_tokens': request.max_new_tokens, 'temperature': request.temperature, 'greedy': request.greedy},
        latency_seconds=time.perf_counter()-start, replacement_characters='\ufffd' in continuation,
        seed=actual_seed, request_id=request_id)
