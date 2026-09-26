# parse_engine.py
# Automated extraction module: maps raw LLM responses to standardized attribute arrays

def extract_attributes(raw_response_text):
    """
    Parses the raw text to extract structural constraints and attributes.
    Returns a dictionary of standardized attributes.
    If parsing fails, returns 'UNRESOLVED'.
    """
    if not raw_response_text or not isinstance(raw_response_text, str):
        return "UNRESOLVED"
    
    try:
        # Simplified parsing logic for illustration matching the text contract
        extracted = {}
        if "chemical" in raw_response_text.lower():
            extracted["domain"] = "Chemical Process Systems"
        elif "architecture" in raw_response_text.lower():
            extracted["domain"] = "Systems Architecture"
        else:
            extracted["domain"] = "Mechanical Systems"
            
        # Mocking regex matching attributes
        extracted["attributes"] = []
        if "tau_s" in raw_response_text:
            extracted["attributes"].append("tau_s")
            
        return extracted
    except Exception:
        return "UNRESOLVED"
