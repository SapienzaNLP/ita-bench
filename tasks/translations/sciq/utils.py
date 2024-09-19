import re

import datasets

QUERY_PREFIX = {
    "en": "Question: {input}\n",
    "it": "Domanda: {input}\n",
}

PASSAGE_PREFIX = {
    "en": "Passage: {passage}\n",
    "it": "Testo: {passage}\n",
}

ANSWER_PREFIX = {
    "en": "Answer: ",
    "it": "Risposta: ",
}


def process_docs(
    dataset: datasets.Dataset,
    source_language: str,
    target_language: str,
    add_passage: bool = False,
) -> datasets.Dataset:
    """
    Prepare the dataset and builds the prompt using the source and target languages.
    """

    def _process_doc(doc):
        if source_language == "en":
            input = doc["input"]
            passage = doc["metadata"]["passage"]
        else:
            input = doc["input_translation"]
            passage = doc["metadata"]["passage_translation"]

        if target_language == "en":
            choices = doc["choices"]
        else:
            choices = doc["choices_translation"]

        query = QUERY_PREFIX[source_language].format(input=input)
        query += ANSWER_PREFIX[target_language]

        if add_passage and passage:
            passage = PASSAGE_PREFIX[source_language].format(passage=passage)
            query = passage + query

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices,
            "gold": int(doc["label"]),
        }

    return dataset.map(_process_doc)


def process_docs_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it")


def process_docs_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en")


def process_docs_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en")


def process_docs_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it")


def process_docs_with_passages_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it", add_passage=True)


def process_docs_with_passages_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en", add_passage=True)


def process_docs_with_passages_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en", add_passage=True)


def process_docs_with_passages_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it", add_passage=True)
