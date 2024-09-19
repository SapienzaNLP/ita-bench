import datasets

QUERY_PREFIX = {
    "default": {
        "en": "Question: {input}",
        "it": "Domanda: {input}",
    },
    "with_choices": {
        "en": "Question: {input}\n{choices}",
        "it": "Domanda: {input}\n{choices}",
    },
}

ANSWER_PREFIX = {
    "en": "Answer:",
    "it": "Risposta:",
}


def process_docs(
    dataset: datasets.Dataset,
    source_language: str,
    target_language: str,
    template="default",
) -> datasets.Dataset:
    """
    Prepare the dataset and builds the prompt using the source and target languages.
    """

    def _process_doc(doc):

        if source_language == "en":
            input = doc["input"]
            choices = doc["choices"]
        else:
            input = doc["input_translation"]
            choices = doc["choices_translation"]

        if template == "with_choices":
            labels = ["A", "B", "C", "D", "E"][: len(choices)]
            choices = [f"* {label}. {choice}" for choice, label in zip(choices, labels)]

            query = QUERY_PREFIX[template][source_language].format(
                input=input,
                choices="\n".join(choices),
            )

        else:
            query = QUERY_PREFIX[template][source_language].format(input=input)

        query += "\n" + ANSWER_PREFIX[target_language]

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices if template == "default" else labels,
            "gold": int(doc["label"]),
        }

    return dataset.map(_process_doc)


# Custom methods to use the ARC-Challenge dataset for perplexity evaluation.
def process_docs_it_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it")


def process_docs_it_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en")


def process_docs_en_en(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en")


def process_docs_en_it(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it")


# Custom methods to use the ARC-Challenge dataset with the choices directly in the prompt a-la MMLU.
def process_docs_it_it_with_choices(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "it", template="with_choices")


def process_docs_it_en_with_choices(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "it", "en", template="with_choices")


def process_docs_en_en_with_choices(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "en", template="with_choices")


def process_docs_en_it_with_choices(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(dataset, "en", "it", template="with_choices")
