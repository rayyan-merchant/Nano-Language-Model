"""Apply the tested independent numeric references to the locked B01 shape."""
import argparse
from pathlib import Path
import torch
from nano_lm.artifacts import atomic_json
from nano_lm.positions import apply_rope
from scripts.final.audit import audit as independent_audit
from scripts.balanced.protocol import configuration, parameter_breakdown, PARAMETERS, provenance


def audit(root, data, device='cpu'):
    result = independent_audit(root, data, device, configuration_fn=configuration,
        parameter_fn=parameter_breakdown, parameters=PARAMETERS, provenance_fn=provenance)
    result['all_six_block_gradient_norms'] = result.pop('all_eight_block_gradient_norms')
    result['experiment'] = 'B01'
    assert len(result['all_six_block_gradient_norms']) == 6
    q=torch.randn(1,6,16,32,device=device)[:,:,0:1].expand(-1,-1,16,-1)
    k=torch.randn_like(q[:,:,0:1]).expand(-1,-1,16,-1)
    qr,kr=apply_rope(q),apply_rope(k)
    torch.testing.assert_close((qr[:,:,5]*kr[:,:,9]).sum(-1),
                              (qr[:,:,7]*kr[:,:,11]).sum(-1),rtol=2e-5,atol=2e-5)
    torch.testing.assert_close((qr[:,:,5]*kr[:,:,9]).sum(-1),
                              (q[:,:,0]*kr[:,:,4]).sum(-1),rtol=2e-5,atol=2e-5)
    result['relative_position_dot_product_and_common_shift_invariance']='passed'
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True);p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu');p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();atomic_json(a.output, audit(a.root,a.dataset,a.device))
