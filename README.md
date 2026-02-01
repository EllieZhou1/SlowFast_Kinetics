# Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition

This repository contains code accompanying our paper [Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](https://arxiv.org/abs/2512.17953), which was presented at two NeurIPS 2025 workshops: (1) What Makes a Good Video: Next Practices in Video Generation and Evaluation, and (2) SPACE in Vision, Language, and Embodied AI.

### Key Components:
* **clip_experiments/action_swap_400.py** – Test CLIP model on action swap dataset
* **siglip/action_swap_400.py** – Test SigLIP2 model on action swap dataset
* **llm_experiments/intern_vl3_2B, 8B, 9B, 38B, and 78B.py** – Test various InternVL3 models on action swap dataset
* **custom_models.py** – Implementation for my four novel classification model architecture solutions
* **dataset_classes folder** - contains custom Dataset classes for each of my models, as each requires different inputs. Does video preprocessing and constructs model-ready tensors and labels from raw frame folders and CSV metadata.
* **flexible_train_test.py** – Centralized training and testing evaluation script which parses yaml config file, builds any one of my model variants, and performs training, validation, and testing.
* **gpt_prompt_tuning/automated_25.py** – Performs automated prompt tuning with GPT, where GPT acts as the prompt engineering, improves the prompt in a feedback loop

