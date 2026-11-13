"""Does our forward pass of ESM C 6B give exactly Biohub's logits and SAE feature values while exposing
every unit's gate, value and activation, every attention head's output and every block's residual
state? What does each kind of access cost?
"""

import torch

from argus import data, net, runlog, sae
from argus.checks import checks


def main():
    log = runlog.start("checks", __doc__)
    m, s = net.load(), sae.load()
    idx, ids = next(data.batches(data.sample(log.input), budget=8192, shard=log.shard))
    ids = ids.cuda()
    n = ids.numel()
    with torch.no_grad():
        log.plain(m, ids)
        log.note(checks=checks(m, s, ids), batch=list(ids.shape))
        nop = lambda l, r: None
        with log.op("forward_to_60", n):
            net.run(m, ids, stop=sae.LAYER)
        with log.op("forward_tapped", n):
            net.logits(m, net.run(m, ids, tap=nop))
        with log.op("forward_with_patterns", n):
            net.logits(m, net.run(m, ids, tap=nop, pattern=True))
    one = ids[:1]
    acts = []
    with log.op("forward_backward_all_units", one.numel()):
        x = net.run(m, one, x=m.esmc.embed(one).requires_grad_(), stop=sae.LAYER, tap=lambda l, r: (r["a"].retain_grad(), acts.append(r["a"])))
        sae.encode(s, x)[0][:, 5, 0].sum().backward()
    log.note(grad_nonzero=float(sum((a.grad != 0).float().mean() for a in acts) / len(acts)))
    log.write()
    print(open(log.path).read())


if __name__ == "__main__":
    main()
