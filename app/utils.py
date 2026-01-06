# app/utils.py
from typing import List

def top_k_unique(seq: List[str], k=3):
    out = []
    for s in seq:
        if s and s not in out:
            out.append(s)
        if len(out) >= k:
            break
    return out
