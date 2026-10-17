# Running on a free Colab T4

These are the exact steps behind the 2026-10-17 run on Colab's free T4 (15 GB, compute capability 7.5, no native bf16).

## 1. Get a T4 runtime

Colab → Runtime → Change runtime type → **T4 GPU** → Save → Connect.

## 2. Setup (one cell)

```python
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
!git clone -q https://github.com/vukrosic/llm-research-kit-t4 /content/llm-research-kit-t4
%cd /content/llm-research-kit-t4
!pip install -q -r requirements.txt 2>&1 | tail -5
```

## 3. Data: 40M-token subset (one cell)

```python
!python -c "from datasets import load_dataset; ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]'); ds.save_to_disk('processed_data/speedrun_40M'); print(ds)"
```

Data downloads straight to the Colab VM and nothing goes through your own computer. `dataset_path="auto"` picks the newest folder in `processed_data/`.

## 4. Train

A Colab cell that runs training blocks and buffers output. Start it in the background instead and read the log:

```python
!cd /content/llm-research-kit-t4 && TQDM_DISABLE=1 nohup python -u train_llm.py > /content/t4run.log 2>&1 &
```

Check progress from any cell, or from the Colab **Terminal** panel (bottom-left icon):

```bash
grep -E 'step [0-9]+ \|' /content/t4run.log | tail -5     # train loss + tok/s every 25 steps
grep 'Val Loss' /content/t4run.log | tail -3                 # eval
grep -A8 'TRAINING RESULTS' /content/t4run.log               # final summary
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

If you want output in the cell instead, use `python -u ... 2>&1 | grep --line-buffered -v httpx | tee /content/t4run.log`, but nohup was more reliable.

### Useful flags

| Flag | Default on T4 | Note |
|---|---|---|
| `--batch_size` | 2 (auto on GPUs < 24 GB) | peak memory was only 5.3 GB at batch 2, so 4–8 should fit (untested) |
| `--train_tokens` | 8,000,000 | e.g. `1000000` for a ~2 min smoke test |
| `--log_every` | 25 | steps between loss/tok/s lines |
| `--eval_every` | config | steps between val evals |
| `--compile` | true | `false` to skip the ~77 s compile |
| `--seed` | 42 | |

## 5. What the log should show

```
🔧 15.x GB GPU (< 24 GB): batch_size defaults to 2 (override with --batch_size)
🔧 bf16 is not native on this GPU: fp32 weights, fp16 autocast, GradScaler
step 25 | loss 7.7792 | 12,xxx tok/s | tokens ... | peak mem 5.32 GB | scale 65536
```

If Muon compile fails you will see `⚠️ Muon compile failed ...; using eager polar express.` and training continues.

## 6. Reference result (2026-10-17, default config, 8M tokens, batch 2, seed 42)

| | |
|---|---|
| Steps | 1,953 |
| Speed | ~12,400 tok/s |
| Peak memory | 5.32 GB allocated (5.95 GB in nvidia-smi), GPU 97% busy |
| Setup + compile | 77 s |
| Active training time | 12 m 31 s |
| Final train loss | 5.066 |
| Final val loss | 4.990 (acc 0.238) |

| Step | Val loss |
|---|---|
| 50 | 7.103 |
| 100 | 6.788 |
| 200 | 6.497 |
| 300 | 6.302 |
| 400 | 6.072 |
| 1953 | 4.990 |

The same run in the original repo (bf16) runs out of memory on the T4. Inductor skips bf16 compilation there, and SDPA falls back to its math kernel. That is why T4 mode exists.

This is a single run and a smoke test, not an experiment. Before claiming anything, compare fp16 with fp32 at the same batch and tokens using two or more seeds, and sweep batch 2/4/8.

## Kaggle (not tested yet)

The steps are the same: Settings → Accelerator → GPU T4 ×2 (one GPU is used) and Internet on. Use `/kaggle/working` instead of `/content`.

## Shutting down

Runtime → Disconnect and delete runtime. Everything under `/content` is lost, including the log, checkpoints and data.
