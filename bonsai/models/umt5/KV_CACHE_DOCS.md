# UMT5 KV-Cache Implementation

## Overview

This document describes the KV-cache implementation for the UMT5 model, which significantly improves inference efficiency during autoregressive text generation.

## What is KV-Cache?

KV-cache (Key-Value cache) is an optimization technique for transformer models during autoregressive generation. Instead of recomputing attention for all previous tokens at each step, we cache the key and value tensors and reuse them. This provides significant speedup, especially for longer sequences.

## Implementation Details

### Cache Structure

The `Cache` dataclass stores:

1. **Decoder Self-Attention Cache:**
   - `decoder_key`: Cached keys for decoder self-attention
   - `decoder_value`: Cached values for decoder self-attention
   - `decoder_position`: Current position in the cache (tracks how many tokens have been generated)

2. **Cross-Attention Cache (Encoder-Decoder):**
   - `encoder_key`: Cached keys from encoder outputs (computed once)
   - `encoder_value`: Cached values from encoder outputs (computed once)

### Key Components

1. **`init_cache()` function**: Initializes cache with appropriate dimensions
2. **`Cache` dataclass**: Stores cached key-value pairs
3. **Modified `UMT5Attention`**: Handles both cached and non-cached forward passes
4. **Updated `generate()` method**: Uses cache for efficient generation

### How It Works

#### For Cross-Attention (Encoder-Decoder):
- Encoder key-values are computed **once** on the first decoder step
- Cached and reused for all subsequent generation steps
- No recomputation needed since encoder outputs don't change

#### For Self-Attention (Decoder):
- Cache is built **incrementally** during generation
- At each step, only the new token's key-values are computed
- New key-values are appended to the cache
- Attention is computed using all cached keys and values

### Performance Benefits

Without cache:
```python
# Each step processes ALL previous tokens
step 1: process 1 token
step 2: process 2 tokens  (token 1 is recomputed)
step 3: process 3 tokens  (tokens 1-2 are recomputed)
...
Total computations: O(n²)
```

With cache:
```python
# Each step processes ONLY the new token
step 1: process 1 token, cache it
step 2: process 1 token, reuse cache
step 3: process 1 token, reuse cache
...
Total computations: O(n)
```

Expected speedup: **2-5x faster** for typical generation lengths (10-50 tokens)

## Usage

### Basic Usage

```python
from bonsai.models.umt5.modeling import UMT5Model
from bonsai.models.umt5.params import create_model, load_model_config

# Load model
model = create_model(UMT5Model, file_dir=checkpoint_path, cfg=config)

# Generate with cache (default)
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    use_cache=True,  # This is the default
)

# Generate without cache (for comparison or debugging)
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
    use_cache=False,
)
```

### Advanced Usage with Manual Cache Control

```python
from bonsai.models.umt5.modeling import init_cache

# Initialize cache manually
batch_size = 1
max_decoder_length = 100
encoder_length = 50

cache = init_cache(
    config=model.config,
    batch_size=batch_size,
    max_decoder_length=max_decoder_length,
    encoder_length=encoder_length,
    dtype=jnp.float32,
)

# Use cache in decoder
decoder_outputs = model.decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_outputs,
    cache=cache,
    use_cache=True,
)
```

## Testing

Run the KV-cache test to verify implementation:

```bash
python -m bonsai.models.umt5.tests.test_kv_cache
```

This test:
1. Generates text with and without cache
2. Compares outputs (should be identical)
3. Measures speedup (should show 2-5x improvement)

## Technical Notes

### Memory Trade-off

KV-cache trades memory for speed:
- **Memory usage**: Increases by storing cache tensors
- **Speed improvement**: 2-5x faster generation

For a model with:
- `num_heads = 6`
- `head_dim = 64`
- `max_length = 512`
- `batch_size = 1`

Cache memory: ~1-2 MB per batch item (relatively small)

### Compatibility

- ✅ Works with all UMT5 model sizes (small, base, xl, xxl)
- ✅ Compatible with JAX JIT compilation
- ✅ Supports batch generation
- ✅ Works with attention masking

### Limitations

- Cache size is pre-allocated based on `max_decoder_length`
- For very long sequences (>1000 tokens), memory usage may become significant
- Currently supports only greedy decoding (beam search can be added in future)

## Implementation Checklist

- [x] Add `Cache` dataclass
- [x] Add `init_cache()` function
- [x] Update `UMT5Attention` to support caching
- [x] Update layer classes (`UMT5LayerSelfAttention`, `UMT5LayerCrossAttention`)
- [x] Update `UMT5Block` to propagate cache
- [x] Update `UMT5Stack` to propagate cache
- [x] Update `generate()` method to use cache
- [x] Add test script
- [x] Update exports in `__all__`
- [x] Add documentation

## Future Improvements

1. **Beam Search Support**: Extend cache to support beam search decoding
2. **Dynamic Cache**: Implement dynamic cache that grows as needed
3. **Cache Compression**: For very long sequences, implement cache compression
4. **Multi-GPU Support**: Optimize cache sharding for multi-device inference

## References

- Original Issue: #137
- T5 Paper: https://arxiv.org/abs/1910.10683
- UMT5 Paper: https://arxiv.org/abs/2304.09151
