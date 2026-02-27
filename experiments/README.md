# LLM Engine Experiments

This directory contains experimental variants of the `llm_engine`, featuring different tokenization strategies.

## Files

- `run_llm.sh`: Experimental version featuring PhO-Compress Stage I (Phonetic recoding + side-channel translation), integrated into `insane_v5_ultra` to attempt higher compression rates mapping graphemes to phonemes deterministically via an initial dictionary layer.
- `run_llm_orig.sh`: The baseline / original implementation (`FINAL`) without the PhO-Compress Stage I mapping applied. 

Both scripts recompile a C++ CUDA engine from source conditionally, and then invoke the resulting binary (`llm_engine`).

## Benchmarks

Both scripts have been augmented to support optional benchmarking metrics to cleanly evaluate performance impacts without perturbing usual workflow. You can enable them with the `--measure` flag. For example: `./run_llm.sh --measure`

**Summary of Single-GPU (2080 Ti) Execution speeds:**

| Metric | llm_engine (ORIG) | llm_engine (PHO) |
| --- | --- | --- |
| **Training (Tokens/sec)**| ~130,000 tok/s | ~131,000 tok/s |
| **Inference Chat (Tokens/sec)**| ~1063 tok/s | ~1050 tok/s |

*Note: Performance on multi-GPU setups currently faces CUDA Graph compilation race conditions or segment limits if executed with concurrent IO pipes. Recommend evaluating speed in isolation or with `--no_graph` for pure kernel throughput evaluations.*

## Usage

To train on your own data, ensure your text file is located either in the current directory or provide an absolute path, and specify `--data <your_file.txt>`. 

### 1. Train on Custom Data

Run the scripts in training mode, specifying your text corpus and the desired output weights file (`--ckpt`):

```bash
# Train using the PHO version
./run_llm.sh --train --measure --data /path/to/your/corpus.txt --ckpt my_custom_model.bin --steps 2000 --batch 64 --seq 128 --gpus 2

# Or train using the original version
./run_llm_orig.sh --train --measure --data /path/to/your/corpus.txt --ckpt my_custom_model.bin --steps 2000 --batch 64 --seq 128 --gpus 2
```
*Note: Wait for this process to complete before testing chat.*

### 2. Run Inference (Chat Mode)
Because these scripts are now designed with a "Quickstart Feature", if your custom model checkpoint (`my_custom_model.bin`) already exists from step 1, the script will **automatically boot directly into Chat Mode**!

So, the next time you want to chat with your model, you just run the exact same command:

```bash
# Chat with the PHO version (auto-detects the saved model)
./run_llm.sh --measure --data /path/to/your/corpus.txt --ckpt my_custom_model.bin

# Chat with the original version (auto-detects the saved model)
./run_llm_orig.sh --measure --data /path/to/your/corpus.txt --ckpt my_custom_model.bin
```

When the `>` prompt appears, type your message and hit Enter. The model will auto-regressively predict the next stream of tokens based on what it learned from your custom corpus!

