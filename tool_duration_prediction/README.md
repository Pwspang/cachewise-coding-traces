# Tool duration predictors

End-to-end TF-IDF + duration-cluster training from parsed traces.

## Pipeline

Tool arguments in release data are redacted, so TF-IDF clusters trained on this data may not offer the same accuracy as the un-sanitized traces. 

## Train

```bash
python release/tool_duration_prediction/train_all.py
```

## Inference

```bash
python release/tool_duration_prediction/infer.py --tool Read --elapsed 5
python release/tool_duration_prediction/infer.py --demo
```
The inference module is integrated into vLLM's KVCache manager to use the light-weight estimators for eviction during online inference. 

