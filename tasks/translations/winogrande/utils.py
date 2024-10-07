import datasets

QUERY_PREFIX = {
    "en": "{input}",
    "it": "{input}",
}

ANSWER_PREFIX = {
    "en": "",
    "it": "",
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

        # Split on the first underscore to get the target text.
        idx = input.index("_")
        input, target_text = input[:idx], input[idx + 1 :]
        input = input.strip() + " "
        target_text = target_text.strip()

        query = QUERY_PREFIX[source_language].format(input=input)
        query += ANSWER_PREFIX[target_language]

        choices = [choice.strip() for choice in choices]
        # TODO: This may not be the best for languages without spaces.
        choices = [choice + " " + target_text for choice in choices]

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices,
            "gold": int(doc["label"]),
        }

    return dataset.map(_process_doc).filter(lambda x: x["query"].strip() != "")


def process_docs_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it")


def process_docs_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en")


def process_docs_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en")


def process_docs_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it")
