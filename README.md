# Lab 2 for KTH course ID2223 Scalable Machine Learning and Deep Learning

For this lab assignment we performed Parameter Efficient Fine-Tuning (PEFT) of a Large Language Model with LoRA on the Finetome dataset. Training was done on Google Colab's T4 GPU supplemented with additional RAM. Periodic checkpointing of the weigths was required due to Colab's usage limits, so that training can be restarted from where it left off. The fine tuned LLM is saved to Hugging Face and converted to a GGUF format since Hugging Face only provides free CPU resources for inference. Finally, this fine tuned LLM is available to be used in our chatbot [here]().

## Task 2
Next we investigated ways to improve pipeline scalability and model performance
(a) model-centric ways to improve the model performance include changing the learning rate, the batch size, or the weight-deacy parameter, or even implementing early stopping. We might also focus on the LoRA configuration and ... \
(b) data-centric ways to improve include using a bigger or better dataset to train on. For one, we found a [deduplicated version of the Finetome 100K dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k-dedup) we used, which is a pretty straight-forward way to get "better" data. Another idea we discussed was using a less generalised dataset of comparable size to create a more specialised chatbot.

## Comparing different foundation LLMS we have finetuned
Since inference is done on CPUs, there is an important trade-off we realised between the size of the model and its performance. For this reason, we decided to primarily focus our comparison on the Llama-3.2 1 billion parameter and 3 billion parameter versions.

To evaluate the performance of the models, we fed both of them the following prompts: 
- Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,
- Describe a tall tower in the capital of France.
maybe more? these two are from the notebook


### unsloth/Llama-3.2-3B-Instruct

### unsloth/Llama-3.2-1B-Instruct
