# GPT-Style Transformer Decoder - Pure NumPy Implementation

implementation of a GPT-style transformer decoder built from scratch using only NumPy. 

## Implemented Components


### ✅ 1. Token Embedding
- Character-level token embeddings
- Maps discrete tokens to continuous vector representations

### ✅ 2. Positional Encoding
- **Sinusoidal encoding** (default): Fixed mathematical positional embeddings
- **Learned encoding** (alternative): Trainable positional embeddings
- Configurable via `pos_encoding` parameter

### ✅ 3. Scaled Dot-Product Attention
- Computes attention scores: `Q @ K^T / sqrt(d_head)`
- Applies softmax for attention weights
- Weighted sum over values (V)

### ✅ 4. Multi-Head Attention
- Separate Q (Query), K (Key), V (Value) projections
- Multiple attention heads running in parallel
- Concatenation of head outputs
- Final linear projection (W_O)

### ✅ 5. Feed-Forward Network (FFN)
- Two-layer fully connected network
- First layer: `d_model → d_ff` (expansion)
- Second layer: `d_ff → d_model` (projection)
- GELU activation function (non-linear)

### ✅ 6. Residual Connections + Layer Normalization
- **Pre-norm architecture**: LayerNorm applied before attention/FFN
- Residual connections: `x = x + sublayer(norm(x))`
- Stabilizes training and enables deeper networks

### ✅ 7. Causal Masking
- Upper-triangular mask prevents attention to future tokens
- Essential for autoregressive generation
- Ensures position `i` only attends to positions `≤ i`

### ✅ 8. Output Layer
- Projection to vocabulary size: `logits = hidden @ E^T`
- Softmax distribution over vocabulary
- Generates probability distribution for next token prediction

## Prerequisites

### Required Software
- Python 3.7+
- NumPy
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Install required packages
pip install numpy jupyter

## How to Run

### Option 1: Run with Jupyter Notebook
```bash
jupyter notebook transformer.ipynb
```

### Option 2: Run with JupyterLab
```bash
jupyter lab transformer.ipynb
```

## Demo Output

The demo code demonstrates:
- Character-level tokenization
- Forward pass through the complete model
- Causal masking verification (upper triangle attention = 0)
- Next-token probability predictions
- Top-5 most likely next tokens for input samples

### Example Output:
```
Logits shape: (2, 16, 29)
Next-token probs shape: (2, 29)
Sum probs per sample: [1. 1.]
Upper-triangle attention sum ~ 0: 0.0
Sample 0 top-5 next tokens: [('e', 0.103), ('g', 0.086), ...]
```

## Model Configuration

Default hyperparameters in the demo:
- **vocab_size**: 29 (lowercase letters + space, period, comma)
- **max_len**: 32 (maximum sequence length)
- **d_model**: 64 (embedding dimension)
- **num_heads**: 4 (attention heads)
- **d_ff**: 256 (FFN hidden dimension)
- **num_layers**: 2 (decoder blocks)
- **pos_encoding**: "sinusoidal" (positional encoding type)

## Code Structure

- **Helper functions**: `softmax()`, `gelu()`
- **TokenEmbedding**: Vocabulary embeddings
- **PositionalEncoding**: Position information
- **LayerNorm**: Layer normalization
- **make_causal_mask()**: Attention masking
- **ScaledDotProductAttention**: Core attention mechanism
- **MultiHeadSelfAttention**: Multi-head wrapper
- **FeedForward**: Position-wise FFN
- **DecoderBlock**: Complete transformer block
- **GPTDecoder**: Full model assembly
- **SimpleCharTokenizer**: Character-level tokenization
