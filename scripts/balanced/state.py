"""Full-state recovery and exact scientific continuation comparisons."""
from pathlib import Path
import torch
from nano_lm.checkpoint import read_checkpoint, restore_training, save_checkpoint
from nano_lm.model import GPT
from nano_lm.config import ModelConfig
from nano_lm.generate import generate
from nano_lm.verification import equal_nested
from scripts.balanced.protocol import MODEL, PARAMETERS

FIELDS = ['model', 'optimizer', 'scheduler', 'scaler', 'rng', 'training_state']
TELEMETRY = {'train_seconds', 'eval_seconds', 'tokens_per_second', 'steps_per_second', 'peak_gpu_bytes'}


def scientific_state(state):
    """Remove only nondeterministic measurement fields, recursively, retaining
    every loss, validation, gradient, sampling, coverage and optimization value."""
    if isinstance(state, dict):
        return {k: scientific_state(v) for k, v in state.items() if k not in TELEMETRY}
    if isinstance(state, list):
        return [scientific_state(v) for v in state]
    return state


def compare_trajectory(a, b):
    exact = {k: equal_nested(a[k], b[k]) for k in FIELDS if k != 'training_state'}
    exact['complete_scientific_training_state'] = equal_nested(scientific_state(a['training_state']), scientific_state(b['training_state']))
    assert all(exact.values()), f'Segmented trajectory changed: {exact}'
    return {'exact': exact, 'raw_training_state_equal': equal_nested(a['training_state'], b['training_state']),
        'excluded_measurement_fields_only': sorted(TELEMETRY),
        'interpretation': 'Wall-clock and memory measurements are not bitwise-equal trajectories; all nontelemetry nested training fields and five checkpoint state fields must be exact.'}


def roundtrip(checkpoint, destination, device):
    payload, tok = read_checkpoint(checkpoint)
    assert payload['config'] == MODEL and payload['metadata']['resolved']['experiment_id'] == 'B01'
    model = GPT(ModelConfig(**payload['config'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = torch.amp.GradScaler('cuda', enabled=False); sampler = torch.Generator()
    restore_training(payload, model, optimizer, scheduler, sampler)
    scaler.load_state_dict(payload['scaler'] or {})
    save_checkpoint(destination, model, tok, optimizer, scheduler, sampler, payload['step'], payload['metadata'], scaler, payload['training_state'])
    actual, _ = read_checkpoint(destination)
    exact = {k: equal_nested(payload[k], actual[k]) for k in FIELDS}
    assert all(exact.values()) and model.parameter_count() == PARAMETERS
    ids = generate(model, torch.tensor([tok.encode('ROMEO:\n')], device=device), 8, greedy=True)
    return dict(status='B01_full_nested_state_roundtrip_passed', step_before=payload['step'],
        step_after=actual['step'], additional_optimizer_updates=0, exact_state=exact,
        parameters=PARAMETERS, device=device, text=tok.decode(ids[0].tolist()))
