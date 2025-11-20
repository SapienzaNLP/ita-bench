import string


def doc_to_text(doc):
    doc_to_text = f"{doc['question']}\n"

    # for i in range(len(doc["options"])):
    #     doc_to_text += f"{string.ascii_uppercase[i]}. {doc['options'][i]}\n"

    # parse option_0 .... option_9
    for i in range(10):
        option_key = f"option_{i}"
        if option_key in doc:
            letter = string.ascii_uppercase[i]
            doc_to_text += f"{letter}. {doc[option_key]}\n"
        else:
            break

    doc_to_text += "Answer:"
    return doc_to_text


def doc_to_choice(doc):
    # return [string.ascii_uppercase[i] for i in range(len(doc["options"]))]
    choices = []
    for i in range(10):
        option_key = f"option_{i}"
        if option_key in doc:
            letter = string.ascii_uppercase[i]
            choices.append(letter)
        else:
            break
    return choices