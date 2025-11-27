import logging
from typing import Dict, List

import datasets


try:
    import re
    import signal

    import sympy
    from math_verify import LatexExtractionConfig, parse, verify
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`math-verify`, `sympy>=1.12`, and antlr4-python3-runtime==4.11 is required for generating translation task prompt templates."
    )


INVALID_ANSWER = "[invalidanswer]"


# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
def doc_to_text(doc: dict) -> str:
    return "Problema:" + "\n" + doc["problem"] + "\n\n" + "Solutione:"


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

    return dataset.map(_process_doc).remove_columns(['problem_translation', 'solution_translation'])


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
    parsed_answer = parse(doc["solution"], extraction_config=[LatexExtractionConfig()])
    if verify(parsed_answer, parsed_candidate):
        retval = 1
    else:
        retval = 0

    try:
        original = process_result_v1(doc, candidates)
    except:  # noqa: E722
        original = 0

    output = {
        "exact_match": retval,
        "exact_match_original": original,
    }
    return output


def process_result_v1(doc: dict, candidates: str) -> int:
    # using the orginal answer extraction method
    unnormalized_answer = get_unnormalized_answer(candidates)
    answer = normalize_final_answer(unnormalized_answer)
    normalized_gold = normalize_final_answer(doc["answer"])
    if answer == INVALID_ANSWER:
        return 0
    if answer.strip() == normalized_gold.strip() or is_equiv(answer, normalized_gold):
        retval = 1
    else:
        retval = 0
    return retval


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


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    eval_logger = logging.getLogger(__name__)
    try:
        with timeout(seconds=1):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS_IT = [
    "quadrato",          # square
    "modi",              # ways
    "interi",            # integers
    "dollari",           # dollars
    "mph",               # invariato (unità)
    "pollici",           # inches
    "ft",                # piedi (abbrev.), lasciato invariato
    "ore",               # hours
    "km",                # invariato
    "unità",             # units
    "\\ldots",           # …
    "sue",               # "sue" (nome proprio?), lasciato invariato
    "punti",             # points
    "piedi",             # feet
    "minuti",            # minutes
    "cifre",             # digits
    "centesimi",         # cents
    "gradi",             # degrees
    "cm",                # invariato
    "g",                 # gm → grammi, abbreviato come "g"
    "libbre",            # pounds
    "metri",             # meters
    "pasti",             # meals
    "spigoli",           # edges
    "studenti",          # students
    "bigliettibambini",  # childrentickets (unito come in originale)
    "multipli",          # multiples
    "\\text{s}",         # invariato
    "\\text{.}",         # invariato
    "\\text{\ns}",       # invariato
    "\\text{}^2",        # invariato
    "\\text{}^3",        # invariato
    "\\text{\n}",        # invariato
    "\\text{}",          # invariato
    r"\mathrm{th}",      # invariato
    r"^\circ",           # °
    r"^{\circ}",         # °
    r"\;",               # ;
    r",\!",              # ,!
    "{,}",               # {,}
    '"',                 # "
    "\\dots",            # …
]



def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
