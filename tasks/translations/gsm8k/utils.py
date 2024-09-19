import datasets

QUERY_PREFIX = {
    "en": "Question: ",
    "it": "Domanda: ",
}

ANSWER_PREFIX = {
    "en": "Answer:",
    "it": "Risposta:",
}


def process_docs(
    dataset: datasets.Dataset,
    source_language: str,
    target_language: str,
    output_type: str = "multiple_choice",
) -> datasets.Dataset:
    """
    Prepare the dataset and builds the prompt using the source and target languages.
    """

    def _process_doc(doc):
        if source_language == "en":
            query = QUERY_PREFIX[source_language] + doc["input"]
        else:
            query = QUERY_PREFIX[source_language] + doc["input_translation"]

        query += "\n" + ANSWER_PREFIX[target_language]
        result = doc["label"]

        if output_type == "multiple_choice":
            choices = [result] + doc["metadata"]["distractors"]

            return {
                "id": doc["id"],
                "query": query,
                "choices": choices,
                "gold": 0,
            }

        else:
            if target_language == "en":
                explanation = doc["metadata"]["explanation"]
            else:
                explanation = doc["metadata"]["explanation_translation"]

            answer = f"{explanation} #### {result}"

            return {
                "id": doc["id"],
                "query": query,
                "answer": answer,
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


def process_docs_with_explanations_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it", output_type="generation")


def process_docs_with_explanations_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en", output_type="generation")


def process_docs_with_explanations_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en", output_type="generation")


def process_docs_with_explanations_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it", output_type="generation")
