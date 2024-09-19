"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import os

import yaml


SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}

SUBJECTS_TRANSLATION = {
    "it": {
        "abstract_algebra": "algebra astratta",
        "anatomy": "anatomia",
        "astronomy": "astronomia",
        "business_ethics": "etica aziendale",
        "clinical_knowledge": "conoscenza clinica",
        "college_biology": "biologia universitaria",
        "college_chemistry": "chimica universitaria",
        "college_computer_science": "informatica universitaria",
        "college_mathematics": "matematica universitaria",
        "college_medicine": "medicina universitaria",
        "college_physics": "fisica universitaria",
        "computer_security": "sicurezza informatica",
        "conceptual_physics": "fisica concettuale",
        "econometrics": "econometria",
        "electrical_engineering": "ingegneria elettrica",
        "elementary_mathematics": "matematica elementare",
        "formal_logic": "logica formale",
        "global_facts": "fatti globali",
        "high_school_biology": "biologia liceale",
        "high_school_chemistry": "chimica liceale",
        "high_school_computer_science": "informatica liceale",
        "high_school_european_history": "storia europea liceale",
        "high_school_geography": "geografia liceale",
        "high_school_government_and_politics": "governo e politica liceale",
        "high_school_macroeconomics": "macroeconomia liceale",
        "high_school_mathematics": "matematica liceale",
        "high_school_microeconomics": "microeconomia liceale",
        "high_school_physics": "fisica liceale",
        "high_school_psychology": "psicologia liceale",
        "high_school_statistics": "statistica liceale",
        "high_school_us_history": "storia degli Stati Uniti liceale",
        "high_school_world_history": "storia del mondo liceale",
        "human_aging": "invecchiamento umano",
        "human_sexuality": "sessualità umana",
        "international_law": "diritto internazionale",
        "jurisprudence": "giurisprudenza",
        "logical_fallacies": "fallacie logiche",
        "machine_learning": "machine learning",
        "management": "management",
        "marketing": "marketing",
        "medical_genetics": "genetica medica",
        "miscellaneous": "varie",
        "moral_disputes": "dispute morali",
        "moral_scenarios": "scenari morali",
        "nutrition": "nutrizione",
        "philosophy": "filosofia",
        "prehistory": "preistoria",
        "professional_accounting": "contabilità professionale",
        "professional_law": "diritto professionale",
        "professional_medicine": "medicina professionale",
        "professional_psychology": "psicologia professionale",
        "public_relations": "relazioni pubbliche",
        "security_studies": "studi sulla sicurezza",
        "sociology": "sociologia",
        "us_foreign_policy": "politica estera degli Stati Uniti",
        "virology": "virologia",
        "world_religions": "religioni mondiali",
    },
}

DESCRIPTIONS = {
    "en": "The following are multiple choice questions (with answers) about {subject}.\n\n",
    "it": "Le seguenti sono domande a scelta multipla (con risposte) su {subject}.\n\n",
}

METHOD_TEMPLATE = """# {subject}
def process_docs_cloze_{subject}(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="{source_language}", target_language="{target_language}", template="cloze", subject="{subject}")

def process_docs_multichoice_{subject}(*args, **kwargs):
    return process_docs(*args, **kwargs, source_language="{source_language}", target_language="{target_language}", template="multichoice", subject="{subject}")

"""


# Create a custom class to represent the function call
class FunctionCall(str):
    pass


# Define a custom representer for the FunctionCall class
def function_representer(dumper, data):
    return dumper.represent_scalar("!function", str(data))


# Register the custom representer for the FunctionCall class
yaml.add_representer(FunctionCall, function_representer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_yaml", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--source_language", type=str, required=True)
    parser.add_argument("--target_language", type=str, required=True)
    parser.add_argument("--base_utils", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    args = parser.parse_args()

    source_language = args.source_language
    target_language = args.target_language
    task_type = args.task_type

    for subject, split in SUBJECTS.items():
        task_name = f"itabench_mmlu_{subject}_{source_language}-{target_language}"
        split_name = f"itabench_mmlu_{split}_{source_language}-{target_language}_tasks"

        if source_language == "en":
            subject_name = subject.replace("_", " ").title()
        else:
            subject_name = SUBJECTS_TRANSLATION[source_language][subject].title()
        description = DESCRIPTIONS[source_language].format(subject=subject_name)

        task_yaml = {
            "include": f"../{args.input_yaml}",
            "task": task_name,
            "tag": [split_name],
            "description": description,
            "process_docs": FunctionCall(f"utils.process_docs_{task_type}_{subject}"),
        }

        output_path = os.path.join(args.output_dir, f"{task_name}.yaml")
        with open(output_path, "w") as f:
            yaml.dump(
                task_yaml,
                f,
                indent=2,
                default_style='"',
                default_flow_style=False,
            )

    with open(args.base_utils, "r") as f:
        base_utils = f.read()

    utils_path = os.path.join(args.output_dir, "utils.py")

    with open(utils_path, "w") as f:
        f.write(base_utils)
        f.write("\n\n")

        for subject in SUBJECTS:
            f.write(
                METHOD_TEMPLATE.format(
                    subject=subject,
                    source_language=source_language,
                    target_language=target_language,
                )
            )

    mmlu_name = f"itabench_mmlu_{source_language}-{target_language}"
    mmlu_yaml = {
        "group": mmlu_name,
        "task": [
            f"itabench_mmlu_{subject}_{source_language}-{target_language}"
            for subject in SUBJECTS
        ],
        "aggregate_metric_list": [
            {
                "metric": "acc",
                "aggregation": "mean",
                "weight_by_size": False,
            },
            {
                "metric": "acc_norm",
                "aggregation": "mean",
                "weight_by_size": False,
            },
        ],
        "metadata": {
            "version": "1.0",
        },
    }

    output_path = os.path.join(args.output_dir, f"_{mmlu_name}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(mmlu_yaml, f, indent=2)


if __name__ == "__main__":
    main()
