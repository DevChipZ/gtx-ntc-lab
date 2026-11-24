# gtx-ntc-lab

Experimental Neural Texture Compression (NTC) benchmarking and VRAM budget planning for GTX GPUs[cite: 2, 4].

## Quick Start
bash
pip install -r requirements.txt
python prototype/ntc_bench.py --size 128 --steps 150 --hidden-dim 32 --save-preview
python prototype/vram_budget.py
python prototype/bundle_bench.py --size 128 --steps 150 --hidden-dim 32


## GTX VRAM Notes

* Keep active texture allocations safely under your GPU's VRAM limit (for example, `3500 MB` on a 4 GB card) to avoid spilling into system RAM and stuttering.
* Prioritize hero/close-up assets over distant/background assets when memory gets tight.
