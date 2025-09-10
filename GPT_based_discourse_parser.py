from openai import OpenAI
from configparser import RawConfigParser
import os
import ast

# Load configuration
config = RawConfigParser()
config.read('config.ini')

# API keys

os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')
client = OpenAI()

# RST Relations (grouped into coarse classes)
RST_RELATIONS = {
    "Subject-Matter": [
        "Elaboration",
        "Elaboration-Additional",
        "Elaboration-General-Specific",
        "Elaboration-Part-Whole",
        "Elaboration-Object-Attribute",
        "Elaboration-Set-Member",
        "Background",
        "Circumstance",
        "Condition",
        "Unconditional",
        "Concession",
        "Evaluation",
        "Interpretation",
        "Means",
        "Manner",
        "Purpose",
        "Restatement",
        "Summary",
        "Otherwise",
        "Preparation"
    ],

    "Presentational": [
        "Antithesis",
        "Concession",   # also fits here depending on annotation
        "Evidence",
        "Justify",
        "Motivation",
        "Enablement"
    ],

    "Textual-Organization": [
        "Contrast",
        "Comparison",
        "Joint",
        "List",
        "Sequence",
        "Same-Unit"
    ],

    "Causal-Temporal": [
        "Cause",
        "Result",
        "Consequence",
        "Condition",
        "Temporal-Before",
        "Temporal-After",
        "Temporal-Same-Time"
    ],

    "Attribution": [
        "Attribution"
    ]
}

# Flattened list of all relations (for validation)
ALL_RST_RELATIONS = sorted({rel for group in RST_RELATIONS.values() for rel in group})

def validate_relation(label: str) -> bool:
    """Check if relation label is valid RST-DT relation."""
    return label in ALL_RST_RELATIONS

print(validate_relation("Elaboration"))   # True
print(validate_relation("Motivation"))    # True
print(validate_relation("RandomLabel"))   # False

def analyze_rst(text: str):
    """
    Sends text to ChatGPT and returns:
      - RST tree as nested dict
      - Satellites that depend on nucleus (cannot stand alone)
    """
    prompt = f"""
You are an RST discourse parser.
Take the following text and segment it into Elementary Discourse Units (EDUs).
Then, construct an RST tree in a Python data structure format (nested dicts).
Also, output a list of EDUs that are satellites which cannot make sense alone.

Text:
{text}

Return strictly valid Python code in this structure:

{{
  "tree": {{
      "edu": <string if nucleus, or None if not>,
      "relation": <string or None>,
      "nucleus": <subtree or None>,
      "satellites": [<list of subtrees>]
  }},
  "dependent_satellites": [<list of EDU strings that cannot stand alone>]
}}
    """



    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )

    # Extract the Python dict from model’s output
    content = response.choices[0].message.content.strip()
    rst_result = eval(content)   # ⚠️ for production, safer to use ast.literal_eval
    return rst_result

def analyze_rst1(text: str):
    """
    Sends text to ChatGPT and returns:
      - RST tree as nested dict (explicit nucleus/satellite structure)
      - Satellites that cannot stand alone
    """
    prompt = f"""
You are an RST discourse parser.
Segment the following text into Elementary Discourse Units (EDUs).
Then build an RST tree in this explicit Python dict format:

{
  "edu": "<text of nucleus EDU or None>",
  "relation": "<RST relation label or None>",
  "nucleus": <subtree or None>,
  "satellites": [
      {
         "edu": "<text of satellite EDU>",
         "relation": "<relation to its nucleus>",
         "nucleus": None,
         "satellites": []
      },
      ...
  ]
}

Rules:
- Each EDU must appear exactly once in the tree.
- Satellites go inside the "satellites" list of their nucleus.
- The "dependent_satellites" list should contain EDUs that cannot stand alone.

Text:
{text}

Return only valid Python code for a dict:
{
  "tree": <the nested RST tree>,
  "dependent_satellites": [<list of EDU strings>]
}
    """

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )

    content = response.choices[0].message.content.strip()

    # Parse safely into a Python dict
    try:
        rst_result = ast.literal_eval(content)
    except Exception as e:
        raise ValueError(f"Could not parse model output: {e}\n\nOutput was:\n{content}")

    return rst_result

if __name__ == "__main__":
    text = (
        "Developed by Peking University's Tangent Lab, "
        "the toolkit is for segmenting Elementary Discourse Units. "
        "It implements an end-to-end neural segmenter based on a neural framework, "
        "addressing data insufficiency by transferring a word representation model trained on a large corpus."
    )

    result = analyze_rst1(text)

    print("RST Tree:")
    print(result["tree"])
    print("\nDependent Satellites (cannot stand alone):")
    for sat in result["dependent_satellites"]:
        print("-", sat)


""" EXPECTED OUTPUT
{
  "tree": {
    "edu": "the toolkit is for segmenting Elementary Discourse Units",
    "relation": None,
    "nucleus": {
        "edu": "It implements an end-to-end neural segmenter based on a neural framework",
        "relation": "Elaboration",
        "nucleus": None,
        "satellites": [
            {
                "edu": "addressing data insufficiency",
                "relation": "Means",
                "nucleus": None,
                "satellites": [
                    {
                        "edu": "by transferring a word representation model trained on a large corpus",
                        "relation": "Means",
                        "nucleus": None,
                        "satellites": []
                    }
                ]
            }
        ]
    },
    "satellites": [
        {
            "edu": "Developed by Peking University's Tangent Lab",
            "relation": "Background",
            "nucleus": None,
            "satellites": []
        }
    ]
  },
  "dependent_satellites": [
    "Developed by Peking University's Tangent Lab",
    "addressing data insufficiency",
    "by transferring a word representation model trained on a large corpus"
  ]
}



"""
