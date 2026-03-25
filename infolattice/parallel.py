import contextlib
import random


@contextlib.contextmanager
def joblib_loader(pbar):
    import joblib

    class CB(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **k):
            pbar.update(1)
            return super().__call__(*a, **k)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = CB
    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        pbar.close()


def map_jobs(jobs, f, *, parallel=False, batch_size=25, n_jobs=-1, loader=False, desc="Computing i_von_neumann"):
    jobs = list(jobs)
    if parallel:
        try:
            import joblib
        except ImportError:
            parallel = False
    if loader:
        try:
            import tqdm
        except ImportError:
            loader = False
    if not parallel:
        for j in jobs:
            yield j, f(*j)
        return
    random.shuffle(jobs)
    batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
    run = lambda b: [(j, f(*j)) for j in b]
    tasks = (joblib.delayed(run)(b) for b in batches)
    if loader:
        with joblib_loader(tqdm.tqdm(total=len(batches), desc=desc)):
            res = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(tasks)
    else:
        res = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(tasks)
    for br in res:
        yield from br
