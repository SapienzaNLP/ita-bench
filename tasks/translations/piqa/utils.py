import re

import datasets

QUERY_PREFIX = {
    "question": {
        "en": "Question: {input}\n",
        "it": "Domanda: {input}\n",
    },
    "text_completion": {
        "en": "{input} ",
        "it": "{input} ",
    },
    "topic": {
        "en": "{input}\n",
        "it": "{input}\n",
    },
    "property": {
        "en": "{input}: ",
        "it": "{input}: ",
    },
}

ANSWER_PREFIX = {
    "question": {
        "en": "Answer: ",
        "it": "Risposta: ",
    },
    "text_completion": {
        "en": "",
        "it": "",
    },
    "topic": {
        "en": "",
        "it": "",
    },
    "property": {
        "en": "",
        "it": "",
    },
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
        category = doc["metadata"]["category"]

        if source_language == "en":
            input = doc["input"]
        else:
            input = doc["input_translation"]

        if target_language == "en":
            choices = doc["choices"]
        else:
            choices = doc["choices_translation"]

        query = QUERY_PREFIX[category][source_language].format(input=input)
        query += ANSWER_PREFIX[category][target_language]

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
