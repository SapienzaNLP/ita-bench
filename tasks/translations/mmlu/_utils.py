import datasets

QUERY_PREFIX = {
    "cloze": {
        "en": "Question: {input}",
        "it": "Domanda: {input}",
    },
    "multichoice": {
        "en": "Question: {input}\n{choices}",
        "it": "Domanda: {input}\n{choices}",
    },
}

ANSWER_PREFIX = {
    "en": "Answer: ",
    "it": "Risposta: ",
}


def process_docs(
    dataset: datasets.Dataset,
    source_language: str,
    target_language: str,
    template="cloze",
    subject=None,
) -> datasets.Dataset:
    """
    Prepare the dataset and builds the prompt using the source and target languages.
    """

    def _process_doc_cloze(doc):
        query = QUERY_PREFIX["cloze"][source_language]

        if source_language == "en":
            input = doc["input"]
        else:
            input = doc["input_translation"]

        query = query.format(input=input)
        query += "\n" + ANSWER_PREFIX[target_language]

        if target_language == "en":
            choices = doc["choices"]
        else:
            choices = doc["choices_translation"]

        return {
            "id": doc["id"],
            "query": query,
            "choices": choices,
            "gold": int(doc["label"]),
        }

    def _process_doc_multichoice(doc):
        query = QUERY_PREFIX["multichoice"][source_language]

        if source_language == "en":
            input = doc["input"]
            choices = doc["choices"]
        else:
            input = doc["input_translation"]
            choices = doc["choices_translation"]

        choices = [c[:-1] if c.endswith(".") else c for c in choices]

        labels = ["A", "B", "C", "D"][: len(choices)]
        choices = [f"{label}. {choice}" for choice, label in zip(choices, labels)]

        query = query.format(input=input, choices="\n".join(choices))
        query += "\n" + ANSWER_PREFIX[target_language]

        return {
            "id": doc["id"],
            "query": query,
            "choices": labels,
            "gold": int(doc["label"]),
        }

    if subject is not None:
        dataset = dataset.filter(
            lambda x: x["metadata"]["subject"] == subject.replace("_", " ")
        )

    if template == "cloze":
        return dataset.map(_process_doc_cloze)
    elif template == "multichoice":
        return dataset.map(_process_doc_multichoice)
    else:
        raise ValueError(f"Unknown template: {template}")
