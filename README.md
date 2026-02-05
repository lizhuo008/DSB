<div align="center">

![logo](asset/logo.png)

<h1>DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs (Paper Coming Soon)</h1>

</div>

Dynamic Sliding Block (DSB) is a training-free block scheduling method. 

DSB Cache is a training-free KV-cache scheme tailored to DSB for diffusion LLMs, further demonstrating the advantages of DSB.

## 🚀 Features
- A better semi-autoregressive paradigm.
- DSB-tailored KV cache.
- A training-free, plug-and-play method, improving quality-speed trade-off.
- Fast inference support for Dream and LLaDA model. 
- Full evaluation provided.

## 🔍 Key Details

![overview](asset/overview.png)

1. **Dynamic Sliding Block (DSB)** is a training-free decoding schedule for diffusion LLMs. Instead of using fixed blocks, it keeps an active block that slides forward and can change its size during inference. This lets the model decode easy/high-confidence tokens earlier (especially near block boundaries) and wait on low-confidence tokens until more context is available—improving the quality–speed trade-off.


2. **DSB Cache** is a training-free KV-cache design built for DSB. Sliding blocks can make newly exposed boundary tokens have unstable (transient) KV states, which hurts caching. To fix this, DSB Cache refreshes a small prefix window before the active block together with the block at every step, while caching the rest. It also does periodic global refreshes to keep the cache consistent—boosting throughput with minimal quality drop.

## 🔧 Installation
### Option A: Quick start (recommended)
```bash
pip install -r requirements.txt
```

### Option B: Reproducible install
```bash
pip install -r requirements-lock.txt
```

## ✨Eval

We provide the eval scripts for the main experiment, you can reproduce it directly. For example:
```bash
cd llada
bash eval_instruct.sh
```
The main result is conducted on an Nvidia H200 140G GPU, we evaluate two variants of DSB: DSB(const.) and DSB (greedy), demonstrating the stable improvement of our method.

![main result](asset/main_result.png)

## 🎓 Citation

Coming Soon...

## 🙏 Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada), [Dream](https://github.com/dream-project/dream) and [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) for their excellent work and open-source contributions.
