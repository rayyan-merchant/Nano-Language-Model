"""Frozen validation stopping controller, separate from gradient mathematics."""
import math
from scripts.balanced.protocol import STOP_POLICY, MAXIMUM


class StoppingRule:
    def __init__(self):
        self.records = []
        self.reference = None
        self.counter = 0
        self.best = None
        self.reason = None

    def observe(self, step: int, bpb: float) -> dict:
        if self.reason is not None:
            raise ValueError('No validation or update is permitted after stopping')
        expected = (len(self.records)+1)*STOP_POLICY['interval']
        if step != expected or step > MAXIMUM or not math.isfinite(bpb) or bpb < 0:
            raise ValueError('Invalid, missing, repeated or nonfinite scheduled validation')
        before = self.reference
        meaningful = before is None or bpb < before - STOP_POLICY['minimum_improvement']
        if meaningful:
            self.reference = bpb; self.counter = 0
        else:
            self.counter += 1
        if self.best is None or bpb < self.best['bpb']:
            self.best = {'step': step, 'bpb': bpb}
        if step >= STOP_POLICY['minimum_updates'] and self.counter >= STOP_POLICY['patience']:
            self.reason = 'early_stop'
        elif step == MAXIMUM:
            self.reason = 'budget_exhausted'
        record = dict(step=step, bpb=bpb, patience_reference_before=before,
                      patience_reference_after=self.reference, meaningful_improvement=meaningful,
                      counter=self.counter, absolute_best=dict(self.best), stopping_reason=self.reason)
        self.records.append(record)
        return record


def replay(history):
    rule = StoppingRule()
    for row in history:
        if 'validation' in row:
            rule.observe(row['step'], row['validation']['bpb'])
    return rule
