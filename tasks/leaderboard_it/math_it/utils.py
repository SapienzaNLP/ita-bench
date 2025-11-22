from typing import Dict, List

import datasets


try:
    from math_verify import LatexExtractionConfig, parse, verify
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`math-verify` is required for generating translation task prompt templates. \
please install math-verify via pip install lm-eval[math] or pip install -e .[math]",
    )


INVALID_ANSWER = "[invalidanswer]"


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problema:" + "\n" + doc["problem"] + "\n\n" + "Soluzione:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["problem_translation"],
            "solution": doc["solution_translation"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution_translation"])),
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc

    return dataset.map(_process_doc)


# def list_fewshot_samples() -> list[dict]:
#     return [
#         {
#             "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
#             "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
#             "few_shot": "1",
#         },
#         {
#             "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
#             "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
#             "few_shot": "1",
#         },
#         {
#             "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
#             "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
#             "few_shot": "1",
#         },
#         {
#             "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
#             "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
#             "few_shot": "1",
#         },
#     ]

def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Trova il dominio dell'espressione $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "Le espressioni all'interno di ciascuna radice quadrata devono essere non negative. Pertanto, $x-2 \\ge 0$, quindi $x\\ge 2$, e $5 - x \\ge 0$, quindi $x \\le 5$. Inoltre, il denominatore non può essere uguale a zero, quindi $5-x>0$, il che dà $x<5$. Pertanto, il dominio dell'espressione è $\\boxed{[2,5)}$.\nRisposta finale: la risposta finale è $[2,5)$. Spero sia corretta.",
            "few_shot": "1",
        },
        {
            "problem": "Se $\\det \\mathbf{A} = 2$ e $\\det \\mathbf{B} = 12$, trova $\\det (\\mathbf{A} \\mathbf{B})$.",
            "solution": "Abbiamo che $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nRisposta finale: la risposta finale è $24$. Spero sia corretta.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell di solito solleva due pesi da 20 libbre per 12 volte. Se usa due pesi da 15 libbre, quante volte deve sollevarli per sollevare lo stesso peso totale?",
            "solution": "Se Terrell solleva due pesi da 20 libbre per 12 volte, solleva un totale di $2\\cdot 12\\cdot20=480$ libbre. Se invece solleva due pesi da 15 libbre per $n$ volte, solleverà un totale di $2\\cdot15\\cdot n=30n$ libbre. Eguagliando a 480 libbre, possiamo risolvere per $n$:\n\\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nRisposta finale: la risposta finale è $16$. Spero sia corretta.",
            "few_shot": "1",
        },
        {
            "problem": "Se il sistema di equazioni\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}\nha una soluzione $(x, y)$ con $x$ e $y$ entrambi non nulli, trova $\\frac{a}{b}$, assumendo che $b$ sia non nullo.",
            "solution": "Se moltiplichiamo la prima equazione per $-\\frac{3}{2}$, otteniamo\n\n$$6y-9x=-\\frac{3}{2}a.$$ Poiché sappiamo anche che $6y-9x=b$, abbiamo\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nRisposta finale: la risposta finale è $-\\frac{2}{3}$. Spero sia corretta.",
            "few_shot": "1",
        },
    ]


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]
    parsed_candidate = parse(candidates)
    parsed_answer = parse(doc["solution_translation"], extraction_config=[LatexExtractionConfig()])
    if verify(parsed_answer, parsed_candidate):
        retval = 1
    else:
        retval = 0

    output = {
        "exact_match": retval,
    }
    return output


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return INVALID_ANSWER

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = INVALID_ANSWER
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    try:
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except AssertionError:
        return INVALID_ANSWER
