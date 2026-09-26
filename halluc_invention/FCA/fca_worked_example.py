"""Exact finite FCA demonstration for revised manuscript Section 10.
Python 3 standard library only. This is not the Hall2Invent benchmark engine.
Run: python3 Code/fca_worked_example.py > Code/fca_example_results.json
"""
from fractions import Fraction
from itertools import combinations
import json

M = frozenset('abcde')
OBJECTS = {'K1': frozenset('abc'), 'K2': frozenset('abd'), 'K3': frozenset('ae')}

def powerset(items):
    xs = sorted(items)
    return [frozenset(c) for n in range(len(xs)+1) for c in combinations(xs,n)]

def extent(b, objects=OBJECTS):
    return tuple(k for k,v in objects.items() if b <= v)

def closure(b, objects=OBJECTS, universe=M):
    a=extent(b, objects)
    return frozenset.intersection(*(objects[k] for k in a)) if a else universe

def concepts(objects=OBJECTS, universe=M):
    return [(extent(b,objects),b) for b in powerset(universe) if closure(b,objects,universe)==b]

def jaccard(x,y):
    return Fraction(len(x&y),len(x|y)) if x|y else Fraction(1)

def novelty(b, objects=OBJECTS, universe=M, min_support=1):
    supported=[intent for ext,intent in concepts(objects,universe) if len(ext)>=min_support]
    return 1-max(jaccard(b,c) for c in supported) if supported else None

def anchor(b):
    # First maximize core cardinality, then Jaccard, then smallest stable ID.
    return min(OBJECTS, key=lambda k:(-len(b&OBJECTS[k]),-jaccard(b,OBJECTS[k]),k))

def target(b): return frozenset('ab') <= b

def checks(b):
    return {'not_c_and_d': not frozenset('cd') <= b, 'e_requires_b': 'e' not in b or 'b' in b}

def record(b):
    k=anchor(b); core=b&OBJECTS[k]
    return {'attributes': sorted(b),'extent':extent(b),'closure':sorted(closure(b)),
            'anchor':k,'core':sorted(core),'core_weights':{k:len(b&v) for k,v in OBJECTS.items()},
            'structural_class':'incremental' if len(core)>=2 else 'weakly_anchored',
            'target_pass':target(b),'hard_checks':checks(b),'supported_novelty':str(novelty(b))}

initial=frozenset('abcd'); core=initial&OBJECTS[anchor(initial)]; r0=initial-core
repairs=[]
for r in powerset(M-core):
    b=core|r; feasible=target(b) and all(checks(b).values())
    distance=len(r^r0); quality=int('e' in b)
    repairs.append({'residual':sorted(r),'candidate':sorted(b),'feasible':feasible,
                    'edit_distance':distance,'quality':quality,
                    'objective':distance-2*quality if feasible else None})
best=min((r for r in repairs if r['feasible']),key=lambda r:r['objective'])
repaired=frozenset(best['candidate']); weak=frozenset('ce')
tiny_objects={'U1':frozenset('a'),'U2':frozenset('b')}; tiny_m=frozenset('ab')
uncorrected=1-max(jaccard(tiny_m,b) for _,b in concepts(tiny_objects,tiny_m))
corrected=novelty(tiny_m,tiny_objects,tiny_m)
# Regression checks target the formal counterexamples and all reported arithmetic.
assert len(concepts())==6
assert closure(frozenset('b'))==frozenset('ab')
assert closure(initial)==M and not extent(initial)
assert novelty(initial)==Fraction(1,4)==novelty(repaired)
assert best['residual']==['e'] and best['objective']==0
assert len(initial^repaired)==len(r0^frozenset(best['residual']))==2
assert novelty(weak)==Fraction(2,3)
assert uncorrected==0 and corrected==Fraction(1,2)
# Original classification-gap counterexample: Jaccard-only would select K1.
gap_d=frozenset(range(10)); gap_k1=frozenset(range(2)); gap_k2=frozenset(range(5))|frozenset(range(10,105))
assert len(gap_k2)==100 and jaccard(gap_d,gap_k1)>jaccard(gap_d,gap_k2)
assert len(gap_d&gap_k1)<3<=len(gap_d&gap_k2)
output={'description':'Deterministic toy example, not empirical benchmark results',
        'concepts':[{'extent':a,'intent':sorted(b),'support':len(a)} for a,b in concepts()],
        'initial':record(initial),'residual_search':repairs,'selected_repair':record(repaired),
        'weak_candidate':record(weak),'toy_evidence':{'score':'4/5','threshold':'11/20','passes':True},
        'empty_extent_counterexample':{'unrestricted_novelty':str(uncorrected),'supported_novelty':str(corrected)},
        'classification_gap_counterexample':{'jaccard_K1':str(jaccard(gap_d,gap_k1)),
            'jaccard_K2':str(jaccard(gap_d,gap_k2)),'corrected_anchor':'K2','corrected_class':'incremental'}}
print(json.dumps(output,indent=2))
