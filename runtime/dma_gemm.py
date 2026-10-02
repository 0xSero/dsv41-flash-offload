"""Run busy experts from a DMA slot through the existing reconstruct/GEMM path."""
from copy import copy


def run_batch(D, st, batch, inners, counts, cap, x, token_sorted,
              weight_sorted, limit, out, apply_fat, use_kernel):
    first, last = D['batches'][batch]
    experts = D['h13'].cold[first:last]
    busy = {e for e in experts if counts[e] > cap}
    if not busy:
        return
    staged = list(inners)
    slot = batch % len(st['s13'])
    for row, e in enumerate(experts):
        if e not in busy:
            continue
        entry = dict(inners[e])
        for key, tensor in [('gate', st['s13'][slot][row][0]),
                            ('up', st['s13'][slot][row][1]),
                            ('down', st['s2'][slot][row])]:
            linear = copy(entry[key])
            linear.trellis = tensor
            entry[key] = linear
        staged[e] = entry
    apply_fat(x, token_sorted, weight_sorted, counts, staged, limit, cap, out,
              use_kernel=use_kernel, skip=set(range(len(inners))) - busy)
