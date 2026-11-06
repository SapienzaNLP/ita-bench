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
    mc_type: str = "mc1",
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
            choices = doc["choices"][mc_type]
        else:
            choices = doc["choices_translation"][mc_type]

        query = QUERY_PREFIX[source_language].format(input=input)
        query += ANSWER_PREFIX[target_language]

        gold = doc["label"][mc_type]
        if mc_type == "mc1":
            # mc1 has exactly one true answer. lm-eval reads a list target as the set
            # of correct choice indices, so give it the index of that answer instead
            # of the 0/1 labels.
            gold = list(gold).index(1)

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices,
            "gold": gold,
        }

    return dataset.map(_process_doc)


def process_results_mc2(doc, results):
    lls, is_greedy = zip(*results)

    # Split on the first `0` as everything before it is true (`1`).
    split_idx = list(doc["label"]["mc2"]).index(0)
    # Compute the normalized probability mass for the correct answer.
    ll_true, ll_false = lls[:split_idx], lls[split_idx:]
    p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
    p_true = p_true / (sum(p_true) + sum(p_false))

    return {"acc": sum(p_true)}


def process_docs_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it")


def process_docs_it_it_mc2(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it", "mc2")


def process_docs_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en")


def process_docs_it_en_mc2(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en", "mc2")


def process_docs_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en")


def process_docs_en_en_mc2(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en", "mc2")


def process_docs_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it")


def process_docs_en_it_mc2(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it", "mc2")
