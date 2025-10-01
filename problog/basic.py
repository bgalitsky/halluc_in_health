from problog.program import PrologString
from problog.core import ProbLog
from problog.sdd_formula import SDD

from multiprocessing import Process, Queue
from problog.program import PrologString
from problog.sdd_formula import SDD
from problog import get_evaluatable

def eval_worker(program: PrologString, queue: Queue):
    try:
        result1 = ProbLog.convert(program, SDD).evaluate()
        queue.put(result1)
    except Exception as e:
        queue.put(e)

def evaluate_with_timeout(program: PrologString, timeout: int = 60):
    q = Queue()
    p = Process(target=eval_worker, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return {}  # empty result on timeout
    res = q.get()
    if isinstance(res, Exception):
        raise res
    return res




p = PrologString("""coin(c1). coin(c2).
0.4::heads(C); 0.6::tails(C) :- coin(C).
win :- heads(C).
query(win).
""")

print(ProbLog.convert(p, SDD).evaluate())


p1 = PrologString("""0.54::inflammation(joints(one)).
0.54::inflammation(joints(few)).
0.54::inflammation(joints(both)).
0.54::inflammation(joints(multiple)).
0.54::inflammation(joints(toe)).
0.54::inflammation(joints(knee)).
0.54::inflammation(joints(ankle)).

0.54::inflammation(pain(painful)).
0.54::inflammation(pain(severe)).
0.54::inflammation(pain(throbbing)).
0.54::inflammation(pain(crushing)).
0.54::inflammation(pain(excruciating)).

0.54::inflammation(property(red)).
0.54::inflammation(property(warm)).
0.54::inflammation(property(tender)).
0.54::inflammation(property(swollen)).
0.54::inflammation(property(fever)).

0.54::inflammation(duration(few_days)).
0.54::inflammation(duration(recur)).
0.54::inflammation(duration(prolonged)).

0.78::inflammation_confirmed :- inflammation(joints(_)), inflammation(pain(_)), inflammation(property(_)), inflammation(duration(_)).
0.78::disease(gout) :- inflammation_confirmed.


0.71::joints(toe).
0.76::pain(severe).
0.34::property(fever).
0.78::property(swelling).
0.58::property(redness).
0.57::joints(knee).
0.57::joints(ankle).
0.57::duration(few_days).
0.57::pain(throbbing).

query(disease(gout)).
query(inflammation_confirmed).
""")

if __name__ == "__main__":
    # everything that starts processes must be under this guard
    problog_program = p1

    result = evaluate_with_timeout(problog_program, timeout=10)
    print(result)