# LLM Engine Experiments 🚀

Experimental high-performance LLM training and inference using custom CUDA kernels and FlashAttention.

## Quickstart (The Simplest Run)

If you just want to see it work immediately on the default dataset (`tinyshakespeare.txt`):

```bash
# To start training from scratch:
./run_llm.sh --train

# To chat with the model (automatically uses 'ckpt.bin' if it exists):
./run_llm.sh
```

---

## 💡 Which script should I use?

*   **`run_llm.sh` (Experimental PHO)**: Uses "PhO-Compress" (phonetic recoding). It maps characters to phonetic tokens to attempt higher compression and efficiency.
*   **`run_llm_orig.sh` (Baseline)**: The standard tokenization baseline. Best for comparing against the PHO variant.

---

## 📂 Training on Custom Data

To train on your own text corpus, specify the `--data` and `--ckpt` (checkpoint name) arguments.

```bash
# Example: Train on your own 'dataset.txt'
./run_llm.sh --train --data dataset.txt --ckpt my_model.bin --steps 1000 --gpus 2
```

Once training finishes, you can chat with your new model using the same command (without `--train`):
```bash
./run_llm.sh --data dataset.txt --ckpt my_model.bin
```

---

## 🛡️ Safety Features (New!)

We've added "Rock Solid" protection to prevent accidental data loss:

1.  **Mandatory `--continue`**: To prevent accidental crashes or mismatched training resumes, you must explicitly use the `--continue` flag to load an existing checkpoint for training.
    ```bash
    ./run_llm.sh --train --continue --ckpt my_model.bin
    ```
2.  **Auto-Incrementing Checkpoints**: If you start a *new* training run (`--train`) and your target `--ckpt` file already exists, the engine will **not** overwrite it. Instead, it will automatically save to `ckpt.bin.1`, `ckpt.bin.2`, etc.
3.  **Concurrent Isolation**: You can run multiple instances of these scripts at the same time in the same folder. Each run isolates its compilation in a unique temporary directory.
4.  **Smart Index Caching**: Index files are now named based on your `K1`, `K2`, and `PHO` settings (e.g., `index_v7_k18192_k28192_pho.bin`). If you change these parameters, a new index will be built automatically to prevent stale data runs.
5.  **Multi-Corpus Support**: You can provide multiple dataset files separated by colons (e.g., `--data "code.txt:prose.txt"`), and they will be expanded correctly.
6.  **Download Safety**: The script now verifies that downloads are successful and the resulting file is not empty before starting.

---

## 📊 Performance Metrics

Use the `--measure` flag to see real-time throughput (Tokens/sec).

| Metric | Baseline (ORIG) | Experimental (PHO) |
| :--- | :--- | :--- |
| **Training** | ~130,000 tok/s | **~131,000 tok/s** |
| **Inference Chat** | ~1063 tok/s | ~1050 tok/s |

*Benchmarks recorded on 2080 Ti GPUs.*
