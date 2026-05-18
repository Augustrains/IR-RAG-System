---
library_name: peft
license: other
base_model: /root/autodl-tmp/IR-RAG-System/models/Qwen3-8B
tags:
- base_model:adapter:/root/autodl-tmp/IR-RAG-System/models/Qwen3-8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3_8b_lora_ir_rag_round2_clean
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3_8b_lora_ir_rag_round2_clean

This model is a fine-tuned version of [/root/autodl-tmp/IR-RAG-System/models/Qwen3-8B](https://huggingface.co//root/autodl-tmp/IR-RAG-System/models/Qwen3-8B) on the ir_rag_train_round2_clean_merged dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2157

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.5755        | 0.4422 | 200  | 1.3893          |
| 1.3134        | 0.8845 | 400  | 1.2578          |
| 1.5073        | 1.3250 | 600  | 1.2327          |
| 1.2932        | 1.7673 | 800  | 1.2206          |
| 1.0900        | 2.2078 | 1000 | 1.2185          |
| 1.2160        | 2.6501 | 1200 | 1.2156          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2