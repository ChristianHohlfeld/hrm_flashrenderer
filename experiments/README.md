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

**Train with defaults:**
```bash
./run_llm.sh --train --data ../tinyshakespeare.txt --ckpt ckpt.bin
```

**Evaluate with metrics:**
```bash
./run_llm.sh --measure --train --steps 50
```

