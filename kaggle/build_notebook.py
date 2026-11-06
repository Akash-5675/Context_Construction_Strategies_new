"""Write kaggle/rag_rerun.ipynb. Run from the project root."""

import json

with open("kaggle/run_worker.py", encoding="utf-8") as f:
    WORKER_SOURCE = f.read()

MD, CODE = "markdown", "code"
cells = [
(MD, """# RAG rerun on Kaggle (two GPUs)

Before you start:
1. In the right sidebar, set **Accelerator: GPU T4 x2** and **Internet: On**.
2. Add your uploaded `rag_bundle` dataset through **Add Input**.
3. For a later session, also add the latest version's output through **Add Input**. Add only the latest one.

First run: leave `MODE = "test"` and click **Run All**.
Full run: set `MODE = "full"`, then use **Save Version > Save & Run All**.
"""),
(CODE, """MODE = "test"             # "test" = 50 runs; "full" = all runs
MAX_SECONDS = 8 * 3600    # stops cleanly so Kaggle can save the output
"""),
(CODE, """import glob, os, shutil, subprocess, sys, time, urllib.request, zipfile

WORK = "/kaggle/working/project"
# Kaggle may keep the zip or unpack it into folders: handle both.
bundles = glob.glob("/kaggle/input/**/rag_bundle.zip", recursive=True)
unpacked = glob.glob("/kaggle/input/**/pipeline/generator.py", recursive=True)
assert bundles or unpacked, "Bundle not found: add your dataset with Add Input."
if unpacked:
    source = os.path.dirname(os.path.dirname(unpacked[0]))
    shutil.copytree(source, WORK, dirs_exist_ok=True)
    print("Copied", source)
else:
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(bundles[0]) as z:
        z.extractall(WORK)
    print("Extracted", bundles[0])
for path in ("data/questions.csv", "pipeline/generator.py", "src/experiment_runner.py"):
    assert os.path.exists(os.path.join(WORK, path)), f"Missing {path}"
print("Chunk files:", len(glob.glob(f"{WORK}/data/chunks/*.pkl")),
      "| Index files:", len(glob.glob(f"{WORK}/indexes/*.index")))

with open(f"{WORK}/run_worker.py", "w", encoding="utf-8") as f:
    f.write(WORKER_SOURCE)

results = [f"/kaggle/working/all_results_v2_w{i}.csv" for i in range(2)]
for i, path in enumerate(results):
    if not os.path.exists(path):
        previous = glob.glob(f"/kaggle/input/**/all_results_v2_w{i}.csv", recursive=True)
        assert len(previous) <= 1, f"Add only the latest output: found {previous}"
        if previous:
            shutil.copy(previous[0], path)
            print("Resuming from", previous[0])
"""),
(CODE, """!pip install -q rouge-score faiss-cpu python-dotenv
!apt-get install -y -q zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh
"""),
(CODE, """gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
print("\\n".join(gpus))
assert len(gpus) == 2, "Need two GPUs: set Accelerator to GPU T4 x2."

PORTS = [11434, 11435]

def start_server(i):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), OLLAMA_HOST=f"127.0.0.1:{PORTS[i]}",
               OLLAMA_KEEP_ALIVE="-1", OLLAMA_NUM_PARALLEL="1")
    log = open(f"/kaggle/working/ollama_{i}.log", "a")
    server = subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORTS[i]}", timeout=2)
            print(f"Ollama {i} is running on GPU {i}")
            return server
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Ollama {i} did not start: check ollama_{i}.log")

servers = [start_server(0), start_server(1)]
!ollama pull llama3.1:8b
!ollama list
"""),
(CODE, """def start_worker(i, seconds_left):
    env = dict(os.environ, GENERATOR_BACKEND="ollama", CUDA_VISIBLE_DEVICES=str(i),
               OLLAMA_URL=f"http://127.0.0.1:{PORTS[i]}/api/chat")
    limit = ["--max-runs", "25"] if MODE == "test" else ["--max-seconds", str(seconds_left)]
    log = open(f"/kaggle/working/worker_{i}.log", "a")
    return subprocess.Popen([sys.executable, "run_worker.py", str(i), results[i]] + limit,
                            cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT)

def rows_done():
    total = 0
    for path in results:
        if os.path.exists(path):
            with open(path, "rb") as f:
                total += f.read().count(b"__k")
    return total

start = time.time()
workers = [start_worker(0, MAX_SECONDS), start_worker(1, MAX_SECONDS)]
last_report = start
while any(w.poll() is None for w in workers):
    time.sleep(30)
    for i in range(2):
        if servers[i].poll() is not None and workers[i].poll() is None:
            print(f"Ollama {i} stopped: restarting")
            servers[i] = start_server(i)
    if time.time() - last_report >= 600:
        last_report = time.time()
        print(f"{(time.time() - start) / 60:.0f} min | saved rows: {rows_done():,}", flush=True)
        !nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

print(f"Workers finished in {(time.time() - start) / 60:.1f} minutes")
print("Exit codes:", [w.returncode for w in workers])
for i in range(2):
    print(f"--- worker {i} ---")
    !tail -n 4 /kaggle/working/worker_{i}.log | cut -c1-200
"""),
(CODE, """import pandas as pd

for server in servers:
    server.terminate()

parts = [pd.read_csv(path) for path in results if os.path.exists(path)]
df = pd.concat(parts, ignore_index=True).drop_duplicates("run_key")
df.to_csv("/kaggle/working/all_results_v2.csv", index=False)
print(f"Rows: {len(df):,} of 12,000")
print(f"Rows per GPU: {[len(p) for p in parts]}")
print(f"Duplicate run keys across files: {sum(len(p) for p in parts) - len(df)}")
print(f"Zero-latency rows: {(df.latency_ms == 0).sum()}")
print(f"Median seconds per generation: {df.latency_ms.median() / 1000:.2f}")
print(f"Refusal rate: {df.answer.str.contains('nsufficient', na=False).mean():.1%}")
print(df.groupby('condition').size().to_string())
print(df[['question_id', 'condition', 'k', 'answer']].head(5).to_string(max_colwidth=100))
if MODE == "full":
    shutil.rmtree(WORK)  # the output only needs the results and logs
"""),
]

cells[2] = (CODE, "WORKER_SOURCE = " + repr(WORKER_SOURCE) + "\n\n" + cells[2][1])

notebook = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": text.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if kind == CODE else {}),
        }
        for kind, text in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("kaggle/rag_rerun.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("wrote kaggle/rag_rerun.ipynb")
