
def doc_to_choice(doc) -> list[str]:
    """
    Convert a doc to a choice.
    """
    return doc["choice_translations"]


DOC_TO_TEXT = "{narrative}\n\n" "{question}\n\n" "{choices}\n" "Risposta:"


def doc_to_text(doc) -> str:
    """
    Convert a doc to text.
    """
    choices = ""
    for i, choice in enumerate(doc["choice_translations"]):
        choices += f"{i+1} - {choice}\n"

    text = DOC_TO_TEXT.format(
        narrative=doc["narrative_translation"], question=doc["question_translation"], choices=choices
    )

    return text
