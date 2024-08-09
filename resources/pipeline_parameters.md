## Hugging Face `model.generate()` Parameter Cheat Sheet

### Core Parameters

| Parameter | Description | Typical Values |
|---|---|---|
| `input_ids` | Tokenized input sequence | Required |
| `attention_mask` | Mask indicating valid tokens (1) and padding (0) | Required for attention-based models |
| `max_length` | Maximum length of generated text | Integer (e.g., 50) |
| `min_length` | Minimum length of generated text | Integer (e.g., 10) |
| `do_sample` | Whether to sample or use greedy decoding | Boolean (True/False) |
| `num_beams` | Number of beams for beam search | Integer (e.g., 5) |
| `early_stopping` | Stop generation when all beams are finished | Boolean (True/False) |
| `temperature` | Controls randomness (higher values = more random) | Float (e.g., 0.7) |
| `top_k` | Consider only top k tokens at each step | Integer (e.g., 50) |
| `top_p` | Consider tokens with cumulative probability <= top_p | Float (e.g., 0.9) |

### Additional Parameters

| Parameter | Description | Typical Values |
|---|---|---|
| `repetition_penalty` | Penalizes repeated sequences | Float (e.g., 1.2) |
| `length_penalty` | Modifies length penalty | Float (e.g., 1.0) |
| `num_return_sequences` | Number of independent sequences | Integer (e.g., 3) |
| `pad_token_id` | Token used for padding | Integer (model-specific) |
| `bos_token_id` | Beginning-of-sequence token | Integer (model-specific) |
| `eos_token_id` | End-of-sequence token | Integer (model-specific) |

### Parameter Guide

* **Greedy Decoding:** `do_sample=False`
  * Generates the most likely sequence at each step.

* **Beam Search:** `do_sample=False`, `num_beams>1`
  * Explores multiple candidates and selects the best sequence.

* **Sampling:** `do_sample=True`
  * Introduces randomness, controlled by `temperature`.

* **Top-k Sampling:** `do_sample=True`, `top_k` set
  * Considers only the top k most probable tokens at each step.

* **Nucleus Sampling:** `do_sample=True`, `top_p` set
  * Considers tokens with cumulative probability less than or equal to `top_p`.

### Tips for Effective Generation

* Experiment with different parameter combinations to find the best settings for your task.
* Start with default values and gradually adjust parameters.
* Consider using a combination of techniques (e.g., beam search with sampling).
* Evaluate generated text using metrics like perplexity, BLEU, or ROUGE.

**Remember:** The optimal parameters depend on the specific model, task, and desired output.
