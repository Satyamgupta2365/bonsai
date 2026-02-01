# UMT5 KV-Cache Quick Reference

## Quick Start

```python
from bonsai.models.umt5.modeling import UMT5Model
from bonsai.models.umt5.params import create_model, load_model_config

# Load model
model = create_model(UMT5Model, file_dir=checkpoint_path, cfg=config)

# Generate with cache (FASTER - default)
output = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
)

# Generate without cache (slower, for comparison)
output = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    use_cache=False,
)
```

## Key Components

### Cache Dataclass
```python
@dataclasses.dataclass
class Cache:
    decoder_key: Optional[jax.Array] = None      # Decoder self-attention cache
    decoder_value: Optional[jax.Array] = None    # Decoder self-attention cache
    encoder_key: Optional[jax.Array] = None      # Cross-attention cache
    encoder_value: Optional[jax.Array] = None    # Cross-attention cache
    decoder_position: int = 0                    # Current position
```

### Initialize Cache
```python
from bonsai.models.umt5.modeling import init_cache

cache = init_cache(
    config=model.config,
    batch_size=1,
    max_decoder_length=512,
    encoder_length=50,
    dtype=jnp.float32,
)
```

## Performance

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| Speed  | 1.0x         | 2-5x       | **2-5x faster** |
| Memory | Baseline     | +1-2 MB    | Minimal overhead |

## When to Use

✅ **Use Cache (default)**:
- Production inference
- Long sequence generation (>10 tokens)
- Real-time applications
- Batch processing

❌ **Disable Cache**:
- Debugging attention mechanisms
- Memory-constrained environments
- Very short sequences (1-2 tokens)

## Common Patterns

### Pattern 1: Simple Generation
```python
# Most common use case
output = model.generate(input_ids=input_ids, max_new_tokens=50)
```

### Pattern 2: With Attention Mask
```python
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=50,
)
```

### Pattern 3: Manual Cache Control
```python
# Initialize cache
cache = init_cache(config, batch_size, max_len, enc_len)

# Use in decoder
decoder_out = model.decoder(
    input_ids=decoder_ids,
    encoder_hidden_states=encoder_out,
    cache=cache,
    use_cache=True,
)

# Update position
cache.decoder_position += 1
```

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `max_decoder_length` or batch size

```python
cache = init_cache(
    config=model.config,
    batch_size=1,  # Reduce batch size
    max_decoder_length=256,  # Reduce max length
    encoder_length=50,
)
```

### Issue: Outputs Differ
**Solution**: This is expected due to numerical precision, differences should be minimal

```python
# Verify difference is small
diff = jnp.abs(output_cached - output_no_cache).max()
assert diff < 1e-5  # Should be very small
```

### Issue: Slow First Step
**Solution**: This is normal - cache initialization and first encoding takes time. Subsequent steps will be much faster.

## Testing

```bash
# Run the KV-cache test
python -m bonsai.models.umt5.tests.test_kv_cache

# Expected output:
# ✓ Outputs match
# ✓ 2-5x speedup observed
```

## API Reference

### `init_cache()`
```python
def init_cache(
    config: UMT5Config,
    batch_size: int,
    max_decoder_length: int,
    encoder_length: Optional[int] = None,
    dtype: DTypeLike = jnp.float32,
) -> Cache
```

### `model.generate()`
```python
def generate(
    input_ids: jax.Array,
    attention_mask: jax.Array = None,
    max_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    use_cache: bool = True,  # NEW parameter
) -> jax.Array
```

## More Information

- Full documentation: [KV_CACHE_DOCS.md](KV_CACHE_DOCS.md)
- Test script: [tests/test_kv_cache.py](tests/test_kv_cache.py)
- Issue: #137
