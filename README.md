# Transformer Decoder Implementation

This repository contains an implementation of the Transformer Decoder architecture, inspired by Andrej Karpathy's tutorial video [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY). The implementation demonstrates the core principles behind the Transformer Decoder, a key building block in modern language models like GPT.

## Features

* **Self-Attention Mechanism**: Implements scaled dot-product attention for context-aware representations
* **Positional Encoding**: Encodes position information into token embeddings for sequence awareness
* **Feedforward Network**: Fully connected layers for feature transformation
* **Layer Normalization**: Normalizes activations for stable training
* **Residual Connections**: Ensures smooth gradient flow through the network

## Requirements

* Python 3.8+
* PyTorch 1.12+
* NumPy
* Jupyter Notebook

Install dependencies using:
```bash
pip install torch numpy jupyter
```

## Usage

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/kartikeyapandey20/transformer-from-scratch.git
cd transformer-from-scratch
```

2. Start Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `transformer-from-scratch.ipynb` in your browser and run the cells sequentially.

### Customization

You can modify the following hyperparameters in the notebook:

* Embedding size: Dimensionality of token embeddings
* Number of heads: Number of attention heads in the multi-head attention mechanism
* Number of layers: Stacking multiple decoder blocks for deeper networks
* Dropout rate: Probability of dropping activations for regularization

## Code Structure

`transformer-from-scratch.ipynb`: Contains the implementation of the Transformer Decoder, including:
* Multi-Head Self-Attention
* Feedforward Network
* Positional Encoding
* Residual Connections and Layer Normalization

## References

* Andrej Karpathy's tutorial: [YouTube Video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* Vaswani et al., "Attention Is All You Need" (2017): [Paper](https://arxiv.org/abs/1706.03762)

## Acknowledgements

Special thanks to Andrej Karpathy for his insightful video that guided this implementation.
