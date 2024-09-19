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


# abstract_algebra
def process_docs_cloze_abstract_algebra(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="abstract_algebra")

def process_docs_multichoice_abstract_algebra(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="abstract_algebra")

# anatomy
def process_docs_cloze_anatomy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="anatomy")

def process_docs_multichoice_anatomy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="anatomy")

# astronomy
def process_docs_cloze_astronomy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="astronomy")

def process_docs_multichoice_astronomy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="astronomy")

# business_ethics
def process_docs_cloze_business_ethics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="business_ethics")

def process_docs_multichoice_business_ethics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="business_ethics")

# clinical_knowledge
def process_docs_cloze_clinical_knowledge(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="clinical_knowledge")

def process_docs_multichoice_clinical_knowledge(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="clinical_knowledge")

# college_biology
def process_docs_cloze_college_biology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="college_biology")

def process_docs_multichoice_college_biology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="college_biology")

# college_chemistry
def process_docs_cloze_college_chemistry(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="college_chemistry")

def process_docs_multichoice_college_chemistry(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="college_chemistry")

# college_computer_science
def process_docs_cloze_college_computer_science(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="college_computer_science")

def process_docs_multichoice_college_computer_science(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="college_computer_science")

# college_mathematics
def process_docs_cloze_college_mathematics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="college_mathematics")

def process_docs_multichoice_college_mathematics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="college_mathematics")

# college_medicine
def process_docs_cloze_college_medicine(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="college_medicine")

def process_docs_multichoice_college_medicine(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="college_medicine")

# college_physics
def process_docs_cloze_college_physics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="college_physics")

def process_docs_multichoice_college_physics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="college_physics")

# computer_security
def process_docs_cloze_computer_security(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="computer_security")

def process_docs_multichoice_computer_security(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="computer_security")

# conceptual_physics
def process_docs_cloze_conceptual_physics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="conceptual_physics")

def process_docs_multichoice_conceptual_physics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="conceptual_physics")

# econometrics
def process_docs_cloze_econometrics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="econometrics")

def process_docs_multichoice_econometrics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="econometrics")

# electrical_engineering
def process_docs_cloze_electrical_engineering(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="electrical_engineering")

def process_docs_multichoice_electrical_engineering(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="electrical_engineering")

# elementary_mathematics
def process_docs_cloze_elementary_mathematics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="elementary_mathematics")

def process_docs_multichoice_elementary_mathematics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="elementary_mathematics")

# formal_logic
def process_docs_cloze_formal_logic(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="formal_logic")

def process_docs_multichoice_formal_logic(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="formal_logic")

# global_facts
def process_docs_cloze_global_facts(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="global_facts")

def process_docs_multichoice_global_facts(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="global_facts")

# high_school_biology
def process_docs_cloze_high_school_biology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_biology")

def process_docs_multichoice_high_school_biology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_biology")

# high_school_chemistry
def process_docs_cloze_high_school_chemistry(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_chemistry")

def process_docs_multichoice_high_school_chemistry(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_chemistry")

# high_school_computer_science
def process_docs_cloze_high_school_computer_science(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_computer_science")

def process_docs_multichoice_high_school_computer_science(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_computer_science")

# high_school_european_history
def process_docs_cloze_high_school_european_history(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_european_history")

def process_docs_multichoice_high_school_european_history(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_european_history")

# high_school_geography
def process_docs_cloze_high_school_geography(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_geography")

def process_docs_multichoice_high_school_geography(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_geography")

# high_school_government_and_politics
def process_docs_cloze_high_school_government_and_politics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_government_and_politics")

def process_docs_multichoice_high_school_government_and_politics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_government_and_politics")

# high_school_macroeconomics
def process_docs_cloze_high_school_macroeconomics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_macroeconomics")

def process_docs_multichoice_high_school_macroeconomics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_macroeconomics")

# high_school_mathematics
def process_docs_cloze_high_school_mathematics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_mathematics")

def process_docs_multichoice_high_school_mathematics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_mathematics")

# high_school_microeconomics
def process_docs_cloze_high_school_microeconomics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_microeconomics")

def process_docs_multichoice_high_school_microeconomics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_microeconomics")

# high_school_physics
def process_docs_cloze_high_school_physics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_physics")

def process_docs_multichoice_high_school_physics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_physics")

# high_school_psychology
def process_docs_cloze_high_school_psychology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_psychology")

def process_docs_multichoice_high_school_psychology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_psychology")

# high_school_statistics
def process_docs_cloze_high_school_statistics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_statistics")

def process_docs_multichoice_high_school_statistics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_statistics")

# high_school_us_history
def process_docs_cloze_high_school_us_history(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_us_history")

def process_docs_multichoice_high_school_us_history(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_us_history")

# high_school_world_history
def process_docs_cloze_high_school_world_history(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="high_school_world_history")

def process_docs_multichoice_high_school_world_history(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="high_school_world_history")

# human_aging
def process_docs_cloze_human_aging(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="human_aging")

def process_docs_multichoice_human_aging(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="human_aging")

# human_sexuality
def process_docs_cloze_human_sexuality(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="human_sexuality")

def process_docs_multichoice_human_sexuality(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="human_sexuality")

# international_law
def process_docs_cloze_international_law(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="international_law")

def process_docs_multichoice_international_law(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="international_law")

# jurisprudence
def process_docs_cloze_jurisprudence(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="jurisprudence")

def process_docs_multichoice_jurisprudence(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="jurisprudence")

# logical_fallacies
def process_docs_cloze_logical_fallacies(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="logical_fallacies")

def process_docs_multichoice_logical_fallacies(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="logical_fallacies")

# machine_learning
def process_docs_cloze_machine_learning(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="machine_learning")

def process_docs_multichoice_machine_learning(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="machine_learning")

# management
def process_docs_cloze_management(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="management")

def process_docs_multichoice_management(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="management")

# marketing
def process_docs_cloze_marketing(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="marketing")

def process_docs_multichoice_marketing(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="marketing")

# medical_genetics
def process_docs_cloze_medical_genetics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="medical_genetics")

def process_docs_multichoice_medical_genetics(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="medical_genetics")

# miscellaneous
def process_docs_cloze_miscellaneous(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="miscellaneous")

def process_docs_multichoice_miscellaneous(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="miscellaneous")

# moral_disputes
def process_docs_cloze_moral_disputes(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="moral_disputes")

def process_docs_multichoice_moral_disputes(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="moral_disputes")

# moral_scenarios
def process_docs_cloze_moral_scenarios(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="moral_scenarios")

def process_docs_multichoice_moral_scenarios(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="moral_scenarios")

# nutrition
def process_docs_cloze_nutrition(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="nutrition")

def process_docs_multichoice_nutrition(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="nutrition")

# philosophy
def process_docs_cloze_philosophy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="philosophy")

def process_docs_multichoice_philosophy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="philosophy")

# prehistory
def process_docs_cloze_prehistory(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="prehistory")

def process_docs_multichoice_prehistory(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="prehistory")

# professional_accounting
def process_docs_cloze_professional_accounting(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="professional_accounting")

def process_docs_multichoice_professional_accounting(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="professional_accounting")

# professional_law
def process_docs_cloze_professional_law(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="professional_law")

def process_docs_multichoice_professional_law(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="professional_law")

# professional_medicine
def process_docs_cloze_professional_medicine(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="professional_medicine")

def process_docs_multichoice_professional_medicine(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="professional_medicine")

# professional_psychology
def process_docs_cloze_professional_psychology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="professional_psychology")

def process_docs_multichoice_professional_psychology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="professional_psychology")

# public_relations
def process_docs_cloze_public_relations(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="public_relations")

def process_docs_multichoice_public_relations(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="public_relations")

# security_studies
def process_docs_cloze_security_studies(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="security_studies")

def process_docs_multichoice_security_studies(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="security_studies")

# sociology
def process_docs_cloze_sociology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="sociology")

def process_docs_multichoice_sociology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="sociology")

# us_foreign_policy
def process_docs_cloze_us_foreign_policy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="us_foreign_policy")

def process_docs_multichoice_us_foreign_policy(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="us_foreign_policy")

# virology
def process_docs_cloze_virology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="virology")

def process_docs_multichoice_virology(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="virology")

# world_religions
def process_docs_cloze_world_religions(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="cloze", subject="world_religions")

def process_docs_multichoice_world_religions(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="en", target_language="en", template="multichoice", subject="world_religions")

