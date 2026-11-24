# gtx-ntc-lab

Experimental Neural Texture Compression (NTC) benchmarking and VRAM budget planning for GTX GPUs.

## Quick Start
## GTX VRAM Notes
```bash
pip install -r requirements.txt
python prototype/ntc_bench.py --size 128 --steps 150 --hidden-dim 32 --save-preview
python prototype/vram_budget.py
python prototype/bundle_bench.py --size 128 --steps 150 --hidden-dim 32
```
