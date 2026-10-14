"""F01 exact state roundtrip and smoke-only continuity; no research updates."""
from pathlib import Path
import torch
from nano_lm.checkpoint import read_checkpoint, restore_training, save_checkpoint
from nano_lm.config import ModelConfig
from nano_lm.model import GPT
from nano_lm.generate import generate
from nano_lm.verification import equal_nested
from scripts.final.protocol import PARAMETERS, MODEL

FIELDS = ['model','optimizer','scheduler','scaler','rng','training_state']


def roundtrip(checkpoint, destination, device):
    payload,tok=read_checkpoint(checkpoint)
    assert payload['config']==MODEL and payload['metadata']['resolved']['experiment_id']=='F01'
    settings=payload['metadata']['resolved']['training']
    assert settings['precision']=='float32' and settings['optimizer']=='AdamW'
    model=GPT(ModelConfig(**payload['config'])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    scaler=torch.amp.GradScaler('cuda',enabled=False);sampler=torch.Generator()
    restore_training(payload,model,optimizer,scheduler,sampler);scaler.load_state_dict(payload['scaler'] or {})
    save_checkpoint(destination,model,tok,optimizer,scheduler,sampler,payload['step'],payload['metadata'],scaler,payload['training_state'])
    actual,_=read_checkpoint(destination)
    exact={key:equal_nested(payload[key],actual[key]) for key in FIELDS}
    assert all(exact.values()) and model.parameter_count()==PARAMETERS
    result=generate(model,torch.tensor([tok.encode('ROMEO:\n')],device=device),8,greedy=True)
    return {'status':'exact_full_state_roundtrip_and_inference_passed','device':device,
            'parameters':PARAMETERS,'step_before':payload['step'],'step_after':actual['step'],
            'additional_optimizer_updates':0,'exact_state':exact,'genuine_text':tok.decode(result[0].tolist())}
