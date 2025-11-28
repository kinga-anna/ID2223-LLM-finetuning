# Lab 2 for KTH course ID2223 Scalable Machine Learning and Deep Learning

For this lab assignment we performed Parameter Efficient Fine-Tuning (PEFT) with LoRA of a Large Language Model on a GPU.

## Task 2
(a) model-centric ways to improve the model performance include changing the learning rate, the batch size, or the weight-deacy parameter, or even implementing early stopping. We could also focus on the LoRA configuration and ... \
(b) data-centric ways to improve include using a bigger or better dataset to train on. For one, we found a [deduplicated version of the Finetome 100K dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k-dedup) we used, which is a pretty straight-forward way to get "better" data. Another idea we discussed was using a less generalised dataset of comparable size to create a more specialised chatbot.

## Comparing different foundation LLMS we have finetuned

### unsloth/Llama-3.2-3B-Instruct
