import streamlit as st
from anytree import Node, RenderTree
import graphviz
import json
from openai import OpenAI
from configparser import RawConfigParser
import os
import ast
from functools import lru_cache



# Load configuration
config = RawConfigParser()
config.read('config.ini')

# API keys

os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')
client = OpenAI()

# ---- Replace this with your real analyze_rst function ----
def example_DT(text: str):
    """
    Dummy RST parser for demo.
    Replace with your actual analyze_rst function that calls ChatGPT.
    """
    return {
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

@st.cache_data(show_spinner=False)
@lru_cache(maxsize=128)
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

{{
  "edu": <text of nucleus EDU or None>,
  "relation": <RST relation label or None>,
  "nucleus": <subtree or None>,
  "satellites": [
      {{
         "edu": <text of satellite EDU>,
         "relation": <relation to its nucleus>,
         "nucleus": None,
         "satellites": []
      }},
      ...
  ]
}}

Rules:
- Each EDU must appear exactly once in the tree.
- Satellites go inside the "satellites" list of their nucleus.
- The "dependent_satellites" list should contain EDUs that cannot stand alone.

Text:
{text}

Return only valid Python code for a dict:
{{
  "tree": <the nested RST tree>,
  "dependent_satellites": [<list of EDU strings>]
}}
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
        #raise ValueError(f"Could not parse model output: {e}\n\nOutput was:\n{content}")
        return content

    return rst_result




# ---- Helper: Convert dict tree into anytree (ASCII) ----
def build_anytree(node_dict, parent=None):
    if not node_dict:
        return None
    label = node_dict.get("edu") or "[None]"
    relation = node_dict.get("relation")
    if relation:
        label += f" ({relation})"
    node = Node(label, parent=parent)
    if node_dict.get("nucleus"):
        build_anytree(node_dict["nucleus"], node)
    for sat in node_dict.get("satellites", []):
        build_anytree(sat, node)
    return node


# ---- Helper: Convert dict tree into Graphviz Digraph ----
def build_graphviz(node_dict, dot=None, parent=None, idx=[0]):
    if dot is None:
        dot = graphviz.Digraph()
    idx[0] += 1
    node_id = str(idx[0])
    label = node_dict.get("edu") or "[None]"
    relation = node_dict.get("relation")
    if relation:
        label += f"\n({relation})"
    dot.node(node_id, label, shape="box", style="rounded,filled", fillcolor="lightgrey")
    if parent:
        dot.edge(parent, node_id)
    if node_dict.get("nucleus"):
        build_graphviz(node_dict["nucleus"], dot, node_id, idx)
    for sat in node_dict.get("satellites", []):
        build_graphviz(sat, dot, node_id, idx)
    return dot


# ---- Recursive UI with expanders ----
def render_tree_ui(node_dict, level=0):
    if not node_dict:
        return
    label = node_dict.get("edu") or "[None]"
    relation = node_dict.get("relation")
    if relation:
        label += f" ({relation})"

    with st.expander(" " * level + label, expanded=(level == 0)):
        if node_dict.get("nucleus"):
            st.markdown("**Nucleus:**")
            render_tree_ui(node_dict["nucleus"], level + 1)

        if node_dict.get("satellites"):
            st.markdown("**Satellites:**")
            for sat in node_dict["satellites"]:
                render_tree_ui(sat, level + 1)

# ---- Streamlit UI ----
st.title("📖 RST Discourse Parser")
st.markdown("[Институт Искусственного Интеллекта МФТИ] (https://iai.mipt.ru/en)")
with st.expander("What is Discourse Analysis and how it is applied?"):
    st.write("""
    Discourse analysis is the study of how language is structured beyond the level of individual sentences. 
    It examines how sequences of sentences, paragraphs, and other textual units work together to create meaning, convey intentions, and guide the reader’s understanding. 
    Unlike traditional grammar analysis, which focuses on syntax and vocabulary, discourse analysis investigates how ideas are connected, how arguments are constructed, and how information is organized to achieve communicative goals.

    At its core, discourse analysis involves breaking down texts into smaller units called **Elementary Discourse Units (EDUs)**. 
    These units typically correspond to clauses or simple sentences that convey a single idea. 
    The relationships between EDUs are then analyzed to understand the **rhetorical structure** of the text. 
    One popular framework for this is **Rhetorical Structure Theory (RST)**, which classifies relations between EDUs as either **nucleus** (central, essential information) or **satellite** (supporting, explanatory, or elaborative information).

    Discourse analysis is applied in a wide range of fields. 
    In **natural language processing (NLP)**, it helps machines understand the structure of texts, enabling tasks such as text summarization, sentiment analysis, and question answering. 
    In **education and linguistics**, it provides insights into how arguments are formed, how narratives flow, and how coherence is maintained in writing. 
    In **business and law**, it can be used to analyze reports, contracts, or legal arguments to ensure clarity and logical consistency.

    By segmenting text into EDUs and mapping the relations between them, discourse analysis allows both humans and machines to capture the deeper meaning of a text, detect the logical flow of information, and identify which parts of the text are essential versus supportive. 
    This structured understanding is crucial for improving communication, automating text understanding, and supporting advanced AI applications.
    """)

st.text("Also fastapi endpoingt is available: http://...:8000/docs")

# ---- Caching wrapper (prevents re-asking GPT for same text) ----
@st.cache_data(show_spinner=False)
def cached_parse(text: str):
    return analyze_rst1(text)

text = st.text_area(
    "Enter text for discourse parsing (should be a few sentences):",
    height=200,
    value="Kirk, 31, who had been invited to speak at Utah Valley University, was seated under a white gazebo addressing a crowd of about 3,000 people in the quad, an outdoor bowl courtyard. According to eyewitnesses and videos taken at the scene, he was responding to a question about gun violence when a single shot rang out around 12:20 local time."
)

if st.button("Parse Discourse"):
    with st.spinner("Parsing ..."):
        result = analyze_rst1(text)

    st.subheader("RST Tree (JSON)")
    st.json(result["tree"])

    st.subheader("Dependent Satellites (cannot stand alone)")
    for sat in result["dependent_satellites"]:
        st.write(f"- {sat}")

    st.subheader("RST Tree (ASCII Visualization)")
    root_node = build_anytree(result["tree"])
    tree_str = ""
    for pre, fill, node in RenderTree(root_node):
        tree_str += f"{pre}{node.name}\n"
    st.code(tree_str, language="text")

    st.subheader("RST Tree (Graphviz Diagram)")
    dot = build_graphviz(result["tree"])
    st.graphviz_chart(dot)
