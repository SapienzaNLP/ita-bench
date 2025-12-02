<p align="center">
  <img src="assets/images/ITA-bench.jpg" />
</p>

# ITA-Bench 🤖🇮🇹
This is the [Sapienza NLP](https://github.com/sapienzanlp) GitHub repository for **ITA-Bench** (Italian Benchmarks), **a benchmark suite for the evaluation of Large Language Models (LLMs) on the Italian language**. ITA-Bench is designed to evaluate the performance of LLMs on a variety of tasks, including question answering, commonsense reasoning, mathematical capabilities, named entity recognition, reading comprehension, and others. 

## Datasets included in ITA-Bench
ITA-Bench includes a variety of datasets for evaluating LLMs on Italian. These datasets are collected from various sources and cover a wide range of tasks.

> [!NOTE]
> All the datasets are available on [🤗 Hugging Face Datasets](https://huggingface.co/collections/sapienzanlp/italian-benchmarks-for-llms-66337ca59e6df7d7d4933896)!

The datasets are divided into three main categories:
1. 🌐 **Translations**: These datasets are translations of existing English datasets into Italian. They are used to evaluate the performance of LLMs on tasks that have been previously studied in the English language, allowing for a direct comparison between models trained on different languages.
    - **Pros**: Translations allow for a direct comparison between models trained on different languages
    - **Cons**: Translations may introduce biases or errors that are not present in the original dataset

2. 🔨 **Adaptations**: These datasets are converted from existing Italian datasets into a format that can be used to evaluate LLMs. They are used to evaluate the performance of LLMs on tasks that may be more specific to the Italian language.
    - **Pros**: The original datasets are already in Italian, so there is no need for translation that may introduce errors
    - **Cons**: These datasets were not originally designed for evaluating LLMs and the adaptation process may introduce biases or errors

3. 🤗 **Leaderboard**: These datasets originate from the archived HuggingFace leaderboard ([LINK](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)). We reintroduce the leaderboard’s tasks in Italian, using existing Italian resources (e.g. MMLU-PROX, and IFEval-ITA) when available, and translating the remaining datasets with an open-source model.


ITA-Bench currently includes the following datasets:
| Dataset | Task | Type | Description |
|---------|------|------|-------------|
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | QA | 🌐 Translation | Commonsense and scientific knowledge |
| [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) | QA | 🌐 Translation | Commonsense and scientific knowledge |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | QA + passage | 🌐 Translation | Boolean questions |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | QA | 🌐 Translation | Simple math word problems |
| [Hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | Completion | 🌐 Translation | Commonsense reasoning |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | QA | 🌐 Translation | Advanced questions on 57 subjects |
| [PIQA](https://huggingface.co/datasets/ybisk/piqa) | QA | 🌐 Translation | Physical interactions reasoning |
| [SciQ](https://huggingface.co/datasets/allenai/sciq) | QA + passage | 🌐 Translation | Scientific reading comprehension |
| [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | QA | 🌐 Translation | Questions on Web misconceptions |
| [WinoGrande](https://huggingface.co/datasets/allenai/winogrande) | Completion | 🌐 Translation | Commonsense reasoning |
| [AMI](https://amievalita2020.github.io/) | QA | 🔨 Adaptation | Misogyny detection |
| [Discotex](https://sites.google.com/view/discotex/home) | Completion | 🔨 Adaptation | Commonsense and world knowledge |
| [Ghigliottinai](https://ghigliottin-ai.github.io/) | QA | 🔨 Adaptation | Guess the missing concept |
| [NERMUD](https://nermud.fbk.eu/) | NER | 🔨 Adaptation | Named entity recognition |
| [PreLearn](https://sites.google.com/view/prelearn20/home) | QA | 🔨 Adaptation | Reasoning about concept relationships |
| [PreTens](https://sites.google.com/view/semeval2022-pretens) | QA | 🔨 Adaptation | Reasoning about concept relationships |
| [QuandHO](https://dh.fbk.eu/2016/03/quandho-question-answering-data-for-italian-history/) | QA | 🔨 Adaptation | Reading comprehension |
| [BBH](https://huggingface.co/datasets/SaylorTwift/bbh) | QA | 🤗 Leaderboard | Hard and complex question on several domains |
| [MATH](https://huggingface.co/datasets/lighteval/MATH-Hard) | QA | 🤗 Leaderboard | Complex mathematical problems |
| [MMLU-PRO](https://huggingface.co/datasets/li-lab/MMLU-ProX) | QA | 🤗 Leaderboard | Advanced questions with 10 possible choices |
| [MUSR](https://huggingface.co/datasets/TAUR-Lab/MuSR) | QA | 🤗 Leaderboard | Multi-step reasoning with commonsense |
| [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | QA | 🤗 Leaderboard | Hard question in biology, physics, and chemistry |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | IF | 🤗 Leaderboard | Instruction Following |



## How to use ITA-Bench
ITA-Bench is designed to be easy to use and flexible. You can evaluate any LLM on the included datasets using the `lm_eval` command-line tool. The tool supports a variety of options to customize the evaluation process, including the ability to specify the LLM model, the number of few-shot examples, and the tasks to evaluate.

### Before you start
We always recommend using a virtual environment to manage your dependencies, e.g., using `venv` or `conda`. To create a new environment with `conda`, you can run:
```bash
# Create a new environment with Conda
conda create -n ita-bench python=3.10

# Always remember to activate the environment before running any command!
conda activate ita-bench
```
> [!NOTE]
> You can read more about managing environments with Conda in the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Evaluating an LLM on ITA-Bench
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
  --tasks itabench_trans_it-it,itabench_adapt_cloze,itabench_adapt_mc,itabench_leaderboard_it \
  --include tasks
```
This command will evaluate `meta-llama/Meta-Llama-3.1-8B-Instruct` on all the benchmarks in our suite. The results will be saved in the `outputs/` directory.

#### Running the evaluation on multiple GPUs
If you have multiple GPUs available, you can use the `accelerate` command to run the evaluation on multiple GPUs:
```bash
accelerate launch -m lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,dtype=bfloat16 \
  --num_fewshot 0 \
  --log_samples \
  --output_path outputs/ \
  --tasks itabench_trans_it-it,itabench_adapt_cloze,itabench_adapt_mc,itabench_leaderboard_it
```

> [!NOTE]
> You can read more about `accelerate` in the [official documentation](https://huggingface.co/docs/accelerate/index).


## Contributing
We welcome contributions to ITA-Bench! 


## License
The code in this repository is licensed under the Apache License, Version 2.0. See the `LICENSE` file for more details.

However, the datasets included in ITA-Bench may have different licenses. Please refer to the original datasets for more information about their licenses.


## Publication and citation
If you use this benchmark or part of it, consider to cite us:

```bibtex
@inproceedings{moroni2024ita,
  title={{ITA-Bench}: Towards a More Comprehensive Evaluation for {I}talian {LLMs}},
  author={Moroni, Luca and Conia, Simone and Martelli, Federico and Navigli, Roberto},
  booktitle={Proceedings of the Tenth Italian Conference on Computational Linguistics (CLiC-it 2024)},
  year={2024},
  url={https://clic2024.ilc.cnr.it/wp-content/uploads/2024/12/66_main_long.pdf},
}
```

## Acknowledgements
* [Future AI Research](https://future-ai-research.it/) for supporting this work.
* [CINECA](https://www.cineca.it/) for providing computational resources.
* [Unbabel](https://unbabel.com/) for building Tower-LLM.
* Thanks to the authors of the original datasets for making them available.
* Thanks to all the [Multilingual Natural Language Processing](http://naviglinlp.blogspot.com/) course students of the Master's of Engineering in Computer Science (Dipartimento di Ingegneria Informatica, Automatica e Gestionale, DIAG) of Sapienza University of Rome for their help in adapting some datasets.
