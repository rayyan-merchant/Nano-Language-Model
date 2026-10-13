"""Replay only the private sampler, never training, to validate exposure lineage."""
import numpy as np
import torch


def replay(payload, training_length=1003854):
    config=payload['config']; settings=payload['metadata']['resolved']['training']
    context=config['context_length']; micro=settings['micro_batch_size']
    accumulation=settings['tokens_per_update']//(context*micro)
    generator=torch.Generator().manual_seed(settings['seed'])
    difference=np.zeros(training_length+1,dtype=np.int64)
    for _ in range(payload['step']*accumulation):
        starts=torch.randint(training_length-context,(micro,),generator=generator).numpy()
        np.add.at(difference,starts+1,1)
        np.add.at(difference,starts+context+1,-1)
    seen=np.cumsum(difference[:-1])>0
    assert torch.equal(generator.get_state(),payload['rng']['sampling'])
    assert np.array_equal(seen,payload['training_state']['seen_bytes'])
    history=payload['training_state']['history']
    assert all(row['accumulation']==accumulation for row in history)
    assert payload['training_state']['training_tokens']==payload['step']*4096
    assert payload['training_state']['training_bytes']==payload['step']*4096
    return {'status':'private_sampling_RNG_and_exact_target_union_replayed',
            'context_length':context,'micro_batch_size':micro,'accumulation':accumulation,
            'sampled_windows_per_update':micro*accumulation,'total_sampled_windows':payload['step']*micro*accumulation,
            'target_positions':payload['step']*4096,'unique_target_bytes':int(seen.sum()),
            'scope':'ASCII canonical character dataset; exact RNG/window union replay, no optimizer updates'}
