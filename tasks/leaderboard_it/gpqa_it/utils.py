import random
import re

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # choices = [
        #     preprocess(doc["Incorrect Answer 1"]),
        #     preprocess(doc["Incorrect Answer 2"]),
        #     preprocess(doc["Incorrect Answer 3"]),
        #     preprocess(doc["Correct Answer"]),
        # ]
        # random.shuffle(choices)
        # correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        choices = doc["choice_translations"]
        correct_answer = choices[doc["label"]]        
        random.shuffle(choices)
        correct_answer_index = choices.index(correct_answer)
        correct_answer_letter = f"({chr(65 + correct_answer_index)})"

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": correct_answer_letter,
        }
        return out_doc

    return dataset.map(_process_doc)
