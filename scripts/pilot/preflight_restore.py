"""Exact restoration/inference of a development checkpoint; no optimizer updates."""
import torch
from nano_lm.checkpoint import read_checkpoint, restore_training, save_checkpoint
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from nano_lm.generate import generate
from nano_lm.verification import equal_nested


def audit_restore(checkpoint, roundtrip, device='cuda'):
    payload,tokenizer=read_checkpoint(checkpoint)
    settings=payload['metadata']['resolved']['training']
    assert settings['run_kind']=='smoke' and settings['optimizer']=='AdamW' and settings['precision']=='float32'
    assert payload['metadata']['resolved']['experiment_id']=='E13' and payload['step']==1
    model=GPT(ModelConfig(**payload['config'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.0)
    scaler=torch.amp.GradScaler(device,enabled=False);generator=torch.Generator()
    restore_training(payload,model,optimizer,scheduler,generator)
    scaler.load_state_dict(payload['scaler'] or {})
    save_checkpoint(roundtrip,model,tokenizer,optimizer,scheduler,generator,1,payload['metadata'],scaler,payload['training_state'])
    actual,_=read_checkpoint(roundtrip)
    exact={k:equal_nested(payload[k],actual[k]) for k in ['model','optimizer','scheduler','scaler','rng','training_state']}
    assert all(exact.values()) and model.parameter_count()==1021824
    ids=torch.tensor([tokenizer.encode('ROMEO:\n')],device=device)
    result=generate(model,ids,4,greedy=True)
    assert result.shape[1]==ids.shape[1]+4
    return {'status':'development_full_state_roundtrip_and_inference_passed','device':device,
            'experiment':'E13','parameters':1021824,'step_before':1,'step_after':1,
            'additional_optimizer_updates':0,'exact_state':exact,'genuine_development_text':tokenizer.decode(result[0].tolist()),
            'scope':'Real-data development checkpoint, not a research measurement or published-output durability claim'}
