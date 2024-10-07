import datasets

PASSAGE_PREFIX = {
    "en": "Context: {passage}",
    "it": "Contesto: {passage}",
}

QUERY_PREFIX = {
    "question": {
        "en": "Question: {input}",
        "it": "Domanda: {input}",
    },
}

ANSWER_PREFIX = {
    "en": "Answer (Yes or No): ",
    "it": "Risposta (Sì o No): ",
}

CHOICES = {
    "en": ["Yes", "No"],
    "it": ["Sì", "No"],
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
        category = doc["metadata"]["category"]

        query = QUERY_PREFIX[category][source_language]

        if source_language == "en":
            query = query.format(input=doc["input"])
        else:
            query = query.format(input=doc["input_translation"])

        if add_passage:
            passage_prefix = PASSAGE_PREFIX[source_language]
            if source_language == "en":
                passage = doc["metadata"]["passage"]
            else:
                passage = doc["metadata"]["passage_translation"]

            if passage:
                passage = passage_prefix.format(passage=passage)
                query = passage + "\n" + query

        query += "\n" + ANSWER_PREFIX[target_language]

        choices = CHOICES[target_language]

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices,
            "gold": 0 if doc["label"] else 1,
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
