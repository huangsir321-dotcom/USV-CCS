"""Reproducible content-derived split; near-duplicate encounter geometry stays together."""
import hashlib
import json
import math
from .dataset import load_case

SPLIT_VERSION='geometry-v1'


def geometry_fingerprint(spec):
    """Relative geometry quantized to 100 m, courses to 5 deg and speeds to .5 m/s.

    Translation and target-row ordering do not affect grouping. Quantization is a
    documented near-duplicate heuristic, not a guarantee for boundary neighbours.
    """
    scenario=load_case(spec)
    own=scenario.own_state
    rows=sorted((round((t.state.x-own.x)/100),round((t.state.y-own.y)/100),
                 round(math.degrees(t.state.psi)%360/5)%72,round(t.state.u/.5)) for t in scenario.targets)
    geometry=(round(math.degrees(own.psi)%360/5)%72,round(own.u/.5),rows)
    return hashlib.sha256(json.dumps(geometry,sort_keys=True).encode()).hexdigest()


def split_cases(specs, seed=7):
    result={'train':[],'val':[],'test':[]}
    for spec in sorted(specs,key=lambda s:s.key):
        key=geometry_fingerprint(spec)
        fraction=int(hashlib.sha256(f'{SPLIT_VERSION}:{seed}:{key}'.encode()).hexdigest()[:16],16)/2**64
        name='train' if fraction<.8 else ('val' if fraction<.9 else 'test')
        result[name].append(spec)
    return result


def split_manifest(splits,seed):
    return {'version':SPLIT_VERSION,'seed':seed,'cases':{k:[s.key for s in v] for k,v in splits.items()},
            'geometry_hashes':{s.key:geometry_fingerprint(s) for v in splits.values() for s in v}}
