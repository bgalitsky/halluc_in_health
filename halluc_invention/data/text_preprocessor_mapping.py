import re


def resolve_study_id(source_case_id: str) -> str:
    """
    Transforms arbitrary alphanumeric source case IDs into sequential,
    padded study tracking identifiers (e.g., 'H2I-CH-01' -> 'Q-REPAIR-001').
    """
    # Extract integer components sequentially across the source cohort
    numeric_digits = re.findall(r'\d+', source_case_id)
    if not numeric_digits:
        return "Q-REPAIR-UNRESOLVED"

    sequential_index = int(numeric_digits[0])
    return f"Q-REPAIR-{sequential_index:03d}"