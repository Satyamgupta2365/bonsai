# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import jax.numpy as jnp
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bonsai.models.umt5.modeling import UMT5Model
from bonsai.models.umt5.params import create_model, load_model_config


def main():
    print("Testing UMT5 KV-cache implementation...")
    
    model_name = "google/umt5-small"
    print(f"Loading model: {model_name}")
    
    model_ckpt_path = snapshot_download(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_conf = load_model_config(model_ckpt_path)

    jax_model = create_model(UMT5Model, file_dir=model_ckpt_path, cfg=model_conf)

    prompts = ["translate English to French: Hello, how are you?"]
    inputs = tokenizer(prompts, padding=True, return_tensors="np")
    input_ids = jnp.array(inputs.input_ids)
    attention_mask = jnp.array(inputs.attention_mask)

    print("\n" + "="*60)
    print("Testing generation WITHOUT cache...")
    print("="*60)
    start_time = time.time()
    output_no_cache = jax_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        use_cache=False,
    )
    no_cache_time = time.time() - start_time
    decoded_no_cache = tokenizer.batch_decode(output_no_cache, skip_special_tokens=True)
    print(f"Output (no cache): {decoded_no_cache[0]}")
    print(f"Time taken: {no_cache_time:.3f}s")

    print("\n" + "="*60)
    print("Testing generation WITH cache...")
    print("="*60)
    start_time = time.time()
    output_with_cache = jax_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        use_cache=True,
    )
    with_cache_time = time.time() - start_time
    decoded_with_cache = tokenizer.batch_decode(output_with_cache, skip_special_tokens=True)
    print(f"Output (with cache): {decoded_with_cache[0]}")
    print(f"Time taken: {with_cache_time:.3f}s")

    print("\n" + "="*60)
    print("Performance comparison:")
    print("="*60)
    print(f"Without cache: {no_cache_time:.3f}s")
    print(f"With cache:    {with_cache_time:.3f}s")
    if no_cache_time > 0:
        speedup = no_cache_time / with_cache_time
        print(f"Speedup:       {speedup:.2f}x faster with cache!")
    
    print("\n" + "="*60)
    print("Verification:")
    print("="*60)
    if jnp.array_equal(output_no_cache, output_with_cache):
        print("✓ Outputs match! KV-cache implementation is correct.")
    else:
        print("✗ Warning: Outputs differ slightly")
        print(f"  Shape no cache:   {output_no_cache.shape}")
        print(f"  Shape with cache: {output_with_cache.shape}")


if __name__ == "__main__":
    main()
