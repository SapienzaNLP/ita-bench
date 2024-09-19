import re

import datasets


def preprocess(text):
    """
    Preprocess the text by removing leading and trailing whitespaces, replacing brackets with periods, and removing any text within brackets.
    Slightly changed from the code from lm_eval/tasks/hellaswag/utils.py.
    """
    text = text.strip()
    text = re.sub("(\\. )?\\[.*?\\]", "\\1", text)
    text = text.replace("  ", " ")
    return text


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
            query = doc["input"]
            activity_label = doc["metadata"]["activity_label"]
        else:
            query = doc["input_translation"]
            activity_label = doc["metadata"]["activity_label_translation"]

        if target_language == "en":
            choices = doc["choices"]
        else:
            choices = doc["choices_translation"]

        query = activity_label + ": " + preprocess(query)
        choices = [preprocess(choice) for choice in choices]

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
