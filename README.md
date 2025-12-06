# Lab 2 for KTH course ID2223 Scalable Machine Learning and Deep Learning

For this lab assignment we performed Parameter Efficient Fine-Tuning (PEFT) of a Large Language Model with LoRA on the Finetome dataset. Training was done on Google Colab's T4 GPU supplemented with additional RAM. Periodic checkpointing of the weigths was required due to Colab's usage limits, so that training can be restarted from where it left off. The fine tuned LLM is saved to Hugging Face and converted to a GGUF format since Hugging Face only provides free CPU resources for inference. Finally, this fine tuned LLM is available to be used in our chatbot [here](https://chatbot-81yytoeishc.streamlit.app/). The source code for the Streamlit app is in a [separate repository](https://github.com/Edwinexd/streamlit-test).

## Task 2
Next we investigated ways to improve pipeline scalability and model performance \
(a) model-centric ways to improve the model performance include changing the learning rate, the batch size, or the weight-deacy parameter, or even implementing early stopping. We might also focus on the LoRA configuration and analyse the impact of different LoRA hyperparameters such as rank, alpha, and dropout. \
(b) data-centric ways to improve include using a bigger or better dataset to train on. For one, we found a [deduplicated version of the Finetome 100K dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k-dedup) we used, which is a pretty straight-forward way to get "better" data. Another idea we discussed was using a less generalised dataset of comparable size to create a more specialised chatbot.

## Comparing different foundation LLMs we have finetuned
Since inference is done on CPUs, there is an important trade-off between the size of the model and its performance. We evaluated three fine-tuned models to determine which provides the best balance of quality and practicality.

To evaluate the performance of the models, we fed all of them the following evaluation prompts:
- Explain quantum computing to a 10-year-old.
- Write a short poem about artificial intelligence.
- What are the main differences between Python and JavaScript?
- How does photosynthesis work?
- Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,
- Describe the water cycle in simple terms.
- What is the capital of France and what is it famous for?
- Explain why the sky is blue.
- Write a haiku about programming.
- What are three tips for learning a new language?

### Evaluation Results

We conducted a voting comparison across all 10 prompts. The results were:

| Model | Base Model | Votes | Percentage |
|-------|------------|-------|------------|
| lora_model_merged | Llama-3.2-3B-Instruct | 7 | 70% |
| lora_model_merged_2 | Llama-3.2-1B-Instruct | 2 | 20% |
| lora_model_merged_3 | Llama-3.2-1B-Instruct (dedup dataset) | 1 | 10% |

### Model Analysis

**Llama-3.2-3B-Instruct (lora_model_merged)**: The 3B model was the clear winner. Our intuition felt its responses were generally more concise and relevant to the prompts while the others tended to go off-topic. One of the best examples of this is comparison of JavaScript and Python where 3B provided correct facts while the others talked about manual vs. automatic memory management which is not differentiating these languages.

**Llama-3.2-1B-Instruct (lora_model_merged_2)**: The 1B model provided serviceable responses but often lacked the depth and polish of the 3B version. It performed adequately on straightforward factual questions but struggled with more creative or complex prompts.

**Llama-3.2-1B-Instruct with dedup dataset (lora_model_merged_3)**: This 1B model was fine-tuned on the deduplicated Finetome dataset. Despite the cleaner training data, it did not outperform the other models in our evaluation. However, since we only tested the dedup dataset on the smaller 1B model, further evaluation of the deduplicated dataset on the 3B model could be worthwhile to isolate the impact of data quality from model capacity.

### Conclusion

Based on our evaluation, we chose the 3B model (lora_model_merged) for our chatbot despite its larger size. The significant quality improvement (70% preference rate) justifies the additional computational requirements for inference and training
