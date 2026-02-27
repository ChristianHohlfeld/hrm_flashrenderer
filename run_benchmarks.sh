#!/bin/bash
rm -f ckpt_pho.bin ckpt_orig.bin
echo "Training PHO..."
env PAIR_K=2048 PAIR_K1=1024 ./run_llm.sh --train --data tinyshakespeare.txt --ckpt ckpt_pho.bin --steps 100 --batch 32 --seq 128 --gpus 2 --measure --log_every 25 > pho_train.log 2>&1
echo "Chat PHO..."
echo "/quit" | env PAIR_K=2048 PAIR_K1=1024 ./run_llm.sh --chat --ckpt ckpt_pho.bin --chat_prompt "Romeo: " --measure > pho_chat.log 2>&1

echo "Training ORIG..."
env PAIR_K=2048 PAIR_K1=1024 ./run_llm_orig.sh --train --data tinyshakespeare.txt --ckpt ckpt_orig.bin --steps 100 --batch 32 --seq 128 --gpus 2 --measure --log_every 25 > orig_train.log 2>&1
echo "Chat ORIG..."
echo "/quit" | env PAIR_K=2048 PAIR_K1=1024 ./run_llm_orig.sh --chat --ckpt ckpt_orig.bin --chat_prompt "Romeo: " --measure > orig_chat.log 2>&1
echo "Done."
