<p align="center">

</p>

Molecular property prediction is central to drug discovery, yet existing frameworks face critical limitations: graph neural networks suffer from over-smoothing, whereas graph Transformers incur quadratic complexity. State-space sequence models enable efficient long-range modeling, but their effectiveness depends critically on molecular graph serialization. Here we present KAN-NC-Mamba, a chemistry-aware framework that integrates selective state-space modeling with nuclear-charge-guided graph serialization. Our node ordering strategy arranges molecular sequences by chemically meaningful atomic semantics derived from periodic electronic structure, providing a coherent prior for state-space propagation. We develop a nuclear-charge-guided Mamba module for linear-complexity long-range dependency capture, and a KAN Dynamic Mixture (KDM) module as a self-attention-enhanced nonlinear fusion architecture for adaptive local-global feature interaction. Experiments on ten benchmarks demonstrate KAN-NC-Mamba's strong performance across classification and regression tasks. Ablation studies reveal that nuclear-charge-guided serialization yields chemically meaningful sequential organization, particularly for polarity-sensitive predictions. This work establishes a chemistry-aware paradigm for integrating graph structures with selective state-space modeling.

<p align="center">

</p>

### Python environment setup with Conda

```bash
conda create --name graph-mamba --file requirements_conda.txt
conda activate graph-mamba
conda clean --all
```
To troubleshoot Mamba installation, please refer to https://github.com/state-spaces/mamba.

For alternative installation via poetry, refer to poetry_steps.txt.

### Running 
```bash
conda activate graph-mamba

# Running Graph-Mamba for bbbp dataset
python main.py --cfg configs/Mamba/ogbg-molbbbp-EX.yaml  wandb.use False
```





