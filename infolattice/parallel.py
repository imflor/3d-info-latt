import contextlib
import heapq
import random
import warnings


VALID_PARALLEL = {"none", "joblib", "slurm"}


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


def normalize_parallel(parallel):
    if parallel is True:
        warnings.warn("Use parallel='joblib' instead of parallel=True.", DeprecationWarning, stacklevel=2)
        return "joblib"
    if parallel is False:
        warnings.warn("Use parallel='none' instead of parallel=False.", DeprecationWarning, stacklevel=2)
        return "none"
    if parallel is None:
        return "none"
    mode = str(parallel).lower()
    if mode not in VALID_PARALLEL:
        raise ValueError(f"parallel must be one of {sorted(VALID_PARALLEL)}")
    return mode


def weighted_chunk_indices(weights, n_chunks):
    weights = [int(w) for w in weights]
    n_chunks = int(n_chunks)
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    chunk_indices = [[] for _ in range(n_chunks)]
    chunk_weights = [0 for _ in range(n_chunks)]
    heap = [(0, chunk_id) for chunk_id in range(n_chunks)]
    heapq.heapify(heap)

    order = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)
    for idx in order:
        total_weight, chunk_id = heapq.heappop(heap)
        chunk_indices[chunk_id].append(idx)
        total_weight += weights[idx]
        chunk_weights[chunk_id] = total_weight
        heapq.heappush(heap, (total_weight, chunk_id))

    return chunk_indices, chunk_weights


def map_jobs(jobs, f, *, parallel="joblib", batch_size=25, n_jobs=-1, loader=False, desc="Computing i_von_neumann"):
    jobs = list(jobs)
    parallel = normalize_parallel(parallel)

    if parallel == "slurm":
        raise RuntimeError("parallel='slurm' does not execute work locally. Use the cluster workflow instead.")

    if parallel == "joblib":
        try:
            import joblib
        except ImportError:
            parallel = "none"

    if loader:
        try:
            import tqdm
        except ImportError:
            loader = False

    if parallel == "none":
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
