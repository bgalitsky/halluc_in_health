from __future__ import annotations
import argparse, csv, json
from typing import Dict, List, Sequence
import numpy as np

EPS = 1e-12

def softmax(theta):
    z = theta - np.max(theta)
    e = np.exp(z)
    return e/np.sum(e)

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))

def f1_score(y, pred):
    tp = np.sum((y==1)&(pred==1))
    fp = np.sum((y==0)&(pred==1))
    fn = np.sum((y==1)&(pred==0))
    p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
    return 2*p*r/max(EPS,p+r)

def split(X,y,frac=.2,seed=13):
    rng=np.random.default_rng(seed)
    idx=np.arange(len(X)); rng.shuffle(idx)
    n=max(1,int(round(len(X)*frac)))
    va=idx[:n]; tr=idx[n:] if len(idx[n:]) else va
    return X[tr],y[tr],X[va],y[va]

def fit_rhetoric_weights(X,y,lr=.08,epochs=3000,l2=1e-4,seed=13):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,size=X.shape[1])
    k=8.0
    for _ in range(epochs):
        w=softmax(theta)
        s=X@w
        p=sigmoid(k*(s-.5))
        grad_s=k*(p-y)/len(y)
        grad_w=X.T@grad_s+l2*w
        J=np.diag(w)-np.outer(w,w)
        theta-=lr*(J@grad_w)
    return softmax(theta)

def choose_tau(X,y,w):
    scores=X@w
    best=(.5,-1)
    for tau in np.linspace(.05,.95,181):
        f=f1_score(y,(scores>=tau).astype(int))
        if f>best[1]: best=(float(tau),float(f))
    return best

def make_pairwise_diffs(rows):
    byq={}
    for r in rows: byq.setdefault(str(r["query_id"]),[]).append(r)
    diffs=[]
    for group in byq.values():
        pos=[r for r in group if int(r["preferred"])==1]
        neg=[r for r in group if int(r["preferred"])==0]
        for p in pos:
            xp=np.array([float(p["explanatory_adequacy"]),
                         float(p["grounding"]),
                         float(p["discourse_weight"])])
            for n in neg:
                xn=np.array([float(n["explanatory_adequacy"]),
                             float(n["grounding"]),
                             float(n["discourse_weight"])])
                diffs.append(xp-xn)
    if not diffs:
        raise ValueError("Need at least one preferred and rejected candidate per query.")
    return np.vstack(diffs)

def fit_validation_weights(diffs,lr=.08,epochs=3000,l2=1e-4,seed=13):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.01,size=diffs.shape[1])
    for _ in range(epochs):
        w=softmax(theta)
        margins=diffs@w
        g=-sigmoid(-margins)/len(margins)
        grad_w=diffs.T@g+l2*w
        J=np.diag(w)-np.outer(w,w)
        theta-=lr*(J@grad_w)
    return softmax(theta)

def fit_lambdas(rows):
    byq={}
    for r in rows: byq.setdefault(str(r["query_id"]),[]).append(r)
    best=(.5,.5,-1)
    grid=np.linspace(0,1,21)
    for lc in grid:
        for ld in grid:
            correct=total=0
            for group in byq.values():
                pos=[r for r in group if int(r["preferred"])==1]
                neg=[r for r in group if int(r["preferred"])==0]
                for p in pos:
                    sp=float(p["goal_fit"])-lc*float(p["contradiction"])-ld*float(p["defeat"])
                    for n in neg:
                        sn=float(n["goal_fit"])-lc*float(n["contradiction"])-ld*float(n["defeat"])
                        correct+=int(sp>sn); total+=1
            acc=correct/max(1,total)
            if acc>best[2]: best=(float(lc),float(ld),float(acc))
    return best

def read_csv(path):
    with open(path,newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rhetoric-csv",required=True)
    ap.add_argument("--candidate-csv",required=True)
    ap.add_argument("--out",default="learned_weights.json")
    ap.add_argument("--seed",type=int,default=13)
    a=ap.parse_args()

    rr=read_csv(a.rhetoric_csv)
    X=np.array([[float(r["relevance"]),float(r["coherence"]),float(r["coverage"])] for r in rr])
    y=np.array([int(r["sufficient"]) for r in rr])
    Xtr,ytr,Xva,yva=split(X,y,seed=a.seed)
    rw=fit_rhetoric_weights(Xtr,ytr,seed=a.seed)
    tau,valf1=choose_tau(Xva,yva,rw)

    cr=read_csv(a.candidate_csv)
    lc,ld,lambda_acc=fit_lambdas(cr)
    denom=1+lc+ld
    for r in cr:
        raw=float(r["goal_fit"])-lc*float(r["contradiction"])-ld*float(r["defeat"])
        r["explanatory_adequacy"]=(raw+lc+ld)/denom

    diffs=make_pairwise_diffs(cr)
    vw=fit_validation_weights(diffs,seed=a.seed)
    rank_acc=float(np.mean((diffs@vw)>0))

    out={
      "rhetoric":{
        "relevance":float(rw[0]),
        "coherence":float(rw[1]),
        "coverage":float(rw[2]),
        "tau":tau,
        "validation_f1":valf1
      },
      "validation":{
        "explanatory_adequacy":float(vw[0]),
        "grounding":float(vw[1]),
        "discourse":float(vw[2]),
        "lambda_contradiction":lc,
        "lambda_defeat":ld,
        "lambda_pairwise_accuracy":lambda_acc,
        "ranking_pairwise_accuracy":rank_acc
      }
    }
    with open(a.out,"w",encoding="utf-8") as f:
        json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
