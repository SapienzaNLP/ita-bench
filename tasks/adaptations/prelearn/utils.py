def doc_to_choice(doc):
    concept_A = doc["concept_A"]
    return [f"non è un prerequisito per {concept_A}", f"è un prerequisito per {concept_A}"]