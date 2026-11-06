import importlib.util
from pathlib import Path

import datasets

REPO = Path(__file__).resolve().parents[1]
UTILS = REPO / "tasks" / "translations" / "truthful_qa" / "utils.py"


def load_utils():
    spec = importlib.util.spec_from_file_location("truthful_qa_utils", UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_doc(doc_id, mc1_labels):
    n = len(mc1_labels)
    return {
        "id": doc_id,
        "input": "Question?",
        "input_translation": "Domanda?",
        "choices": {
            "mc1": [f"answer {i}" for i in range(n)],
            "mc2": ["true", "false"],
        },
        "choices_translation": {
            "mc1": [f"risposta {i}" for i in range(n)],
            "mc2": ["vero", "falso"],
        },
        "label": {"mc1": mc1_labels, "mc2": [1, 0]},
    }


def lm_eval_acc(pred, gold):
    # The rule lm-eval's multiple_choice scoring uses: a list target is a set of
    # correct indices, anything else is a single index.
    return pred in gold if isinstance(gold, list) else pred == gold


def test_mc1_gold_is_the_index_of_the_true_answer():
    utils = load_utils()
    rows = [make_doc("a", [1, 0, 0, 0]), make_doc("b", [0, 0, 1, 0])]
    docs = datasets.Dataset.from_list(rows)
    for process in (
        utils.process_docs_it_it,
        utils.process_docs_it_en,
        utils.process_docs_en_en,
        utils.process_docs_en_it,
    ):
        out = process(docs)
        assert out[0]["gold"] == 0
        assert out[1]["gold"] == 2


def test_mc1_counts_only_the_true_answer_as_correct():
    utils = load_utils()
    docs = datasets.Dataset.from_list([make_doc("a", [1, 0, 0, 0])])
    gold = utils.process_docs_it_it(docs)[0]["gold"]
    scored = [lm_eval_acc(pred, gold) for pred in range(4)]
    assert scored == [True, False, False, False]


def test_mc2_gold_is_unchanged():
    utils = load_utils()
    docs = datasets.Dataset.from_list([make_doc("a", [1, 0, 0, 0])])
    assert utils.process_docs_it_it_mc2(docs)[0]["gold"] == [1, 0]
