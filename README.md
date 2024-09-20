# ITA-Bench 🤖🇮🇹
This is the [Sapienza NLP](https://github.com/sapienzanlp) GitHub repository for ITA-Bench (Italian Benchmarks), **a benchmark suite for the evaluation of Large Language Models (LLMs) on the Italian language**. ITA-Bench is designed to evaluate the performance of LLMs on a variety of tasks, including question answering, commonsense reasoning, mathematical capabilities, named entity recognition, reading comprehension, and others. 

## Datasets included in ITA-Bench
ITA-Bench includes a variety of datasets for evaluating LLMs on Italian. These datasets are collected from various sources and cover a wide range of tasks.

> [!NOTE]
> All the datasets are available on [🤗 Hugging Face Datasets](https://huggingface.co/collections/sapienzanlp/italian-benchmarks-for-llms-66337ca59e6df7d7d4933896)!

The datasets are divided into two main categories:
1. 🌐 **Translations**: These datasets are translations of existing English datasets into Italian. They are used to evaluate the performance of LLMs on tasks that have been previously studied in the English language, allowing for a direct comparison between models trained on different languages.
    - **Pros**: Translations allow for a direct comparison between models trained on different languages
    - **Cons**: Translations may introduce biases or errors that are not present in the original dataset

2. 🔨 **Adaptations**: These datasets are converted from existing Italian datasets into a format that can be used to evaluate LLMs. They are used to evaluate the performance of LLMs on tasks that may be more specific to the Italian language.
    - **Pros**: The original datasets are already in Italian, so there is no need for translation that may introduce errors
    - **Cons**: These datasets were not originally designed for evaluating LLMs and the adaptation process may introduce biases or errors

ITA-Bench currently includes the following datasets:
| Dataset | Task | Type | Description |
|---------|------|------|-------------|
| ARC-Challenge | QA | 🌐 Translation | Commonsense and scientific knowledge |
| ARC-Easy | QA | 🌐 Translation | Commonsense and scientific knowledge |
| BoolQ | QA + passage | 🌐 Translation | Boolean questions |
| GSM8K | QA | 🌐 Translation | Simple math word problems |
| Hellaswag | Completion | 🌐 Translation | Commonsense reasoning |
| MMLU | QA | 🌐 Translation | Advanced questions on 57 subjects |
| PIQA | QA | 🌐 Translation | Physical interactions reasoning |
| SciQ | QA + passage | 🌐 Translation | Scientific reading comprehension |
| TruthfulQA | QA | 🌐 Translation | Questions on Web misconceptions |
| WinoGrande | Completion | 🌐 Translation | Commonsense reasoning |
| AMI | QA | 🔨 Adaptation | Misoginy detection |
| Discotex | Completion | 🔨 Adaptation | Commonsense and world knowledge |
| Ghigliottinai | QA | 🔨 Adaptation | Guess the missing concept |
| NERMUD | NER | 🔨 Adaptation | Named entity recognition |
| PreLearn | QA | 🔨 Adaptation | Reasoning about concept relationships |
| PreTens | QA | 🔨 Adaptation | Reasoning about concept relationships |
| QuandHO | QA | 🔨 Adaptation | Reading comprehesion |
| WiC | QA | 🔨 Adaptation | Word sense disambiguation |


## How to use
To use ITA-Bench, you can follow these steps:
1. Clone this repository:
```bash
git clone git@github.com:SapienzaNLP/ita-bench.git
cd ita-bench
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Run the evaluation script:
```bash
lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=bfloat16 \
  --num_fewshot 0 \
  --log_samples \
  --output_path outputs/ \
  --tasks itabench_trans_it-it,itabench_adapt_cloze,itabench_adapt_mc \
  --include tasks
```
This command will evaluate `meta-llama/Meta-Llama-3.1-8B-Instruct` on all the benchmarks in our suite. The results will be saved in the `outputs/` directory.

## Contributing
We welcome contributions to ITA-Bench! 


## License
The code in this repository is licensed under the Apache License, Version 2.0. See the `LICENSE` file for more details.

However, the datasets included in ITA-Bench may have different licenses. Please refer to the original datasets for more information about their licenses.


## Publication and citation
> Coming soon: a paper on our benchmark suite is under review. Stay tuned for updates!

## Acknowledgements
* [Future AI Research](https://future-ai-research.it/) for supporting this work.
* [CINECA](https://www.cineca.it/) for providing computational resources.
* [Unbabel](https://unbabel.com/) for building Tower-LLM.
* Thanks to the authors of the original datasets for making them available.
