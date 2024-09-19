import datasets
import numpy as np

QUERY_PREFIX = {
    "en": "Question: {input}\n",
    "it": "Domanda: {input}\n",
}

ANSWER_PREFIX = {
    "en": "Answer:",
    "it": "Risposta:",
}


def process_docs(
    dataset: datasets.Dataset,
    source_language: str,
    target_language: str,
) -> datasets.Dataset:
    """
    Prepare the dataset and builds the prompt using the source and target languages.
    """

    def _process_doc(doc):
        if source_language == "en":
            input = doc["input"]
        else:
            input = doc["input_translation"]

        if target_language == "en":
            choices = doc["choices"]
        else:
            choices = doc["choices_translation"]

        query = QUERY_PREFIX[source_language].format(input=input)
        query += ANSWER_PREFIX[target_language]

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices,
            "gold": doc["label"],
        }

    return dataset.map(_process_doc)


def process_results_mc2(doc, results):
    lls, is_greedy = zip(*results)

    # Split on the first `0` as everything before it is true (`1`).
    split_idx = list(doc["label"]).index(0)
    # Compute the normalized probability mass for the correct answer.
    ll_true, ll_false = lls[:split_idx], lls[split_idx:]
    p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
    p_true = p_true / (sum(p_true) + sum(p_false))

    return {"acc": sum(p_true)}


def process_docs_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it")


def process_docs_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en")


def process_docs_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en")


def process_docs_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it")
