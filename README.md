# Tool Series for LLM Verification
## 1. Motivation & Background  
Large Language Models (LLMs) generate fluent and convincing text, but when applied in high-stakes domains such as healthcare, they can produce **hallucinations**, logical inconsistencies, or reasoning errors. To improve trustworthiness, we propose a **neuro-symbolic** pipeline that combines discourse structure analysis with logic-programming (defeasible, probabilistic, abductive) and argumentation techniques to **verify**, **flag**, and **repair** LLM outputs.

This approach builds on the “Tool Series for LLM Verification” framework:  
- We parse the **discourse structure** of the generated or input text (e.g., rhetorical relations such as evidence→claim, cause→effect) to assess coherence. :contentReference[oaicite:0]{index=0}  
- We run the extracted claims through an ontology / logic-program runner to test consistency with domain knowledge (medical, diagnostic, etc.). :contentReference[oaicite:1]{index=1}  
- We apply a **rule-attenuation engine** to model the weakening of conditions in uncertain real-world reasoning. :contentReference[oaicite:2]{index=2}  
- We incorporate **argumentation frameworks** and **probabilistic logic** to address conflicting claims, missing evidence, and degrees of belief. :contentReference[oaicite:3]{index=3}  
- The end-to-end pipeline moves from discourse parsing → logic verification → diagnosis-making verifier, delivering a “truth meter” for LLM answers (✅ verified vs ❌ unreliable). :contentReference[oaicite:4]{index=4}  

Our repository implements these ideas in a health/diagnosis context, where LLM-generated text (e.g., differential diagnoses, patient complaints, treatment suggestions) is subjected to rigorous verification.

---

## 2. What This Repository Contains  
- **Python modules** for preprocessing, discourse parsing, and LLM output ingestion (e.g., `GPT_based_discourse_parser.py`, `dis_sent_discourse_parser.py`).  
- **Logic programming folders** implementing Prolog/ProbLog representations of medical ontologies (see `prolog/`, `problog/`).  
- **Diagnosis modules** (`diagnosis_predictor.py`, `diagnosis_maker.py`) that integrate detected discourse structure with logic and argumentation layers to assess or revise LLM-generated diagnoses.  
- **Data sets** and sample inputs in the `data/` folder (including patient complaint simulants, discourse-annotated text, logic templates).  
- **Visualization and exploratory tools** (`viz/`) for demonstrating how discourse and logic layers interact.  
- **Quick-start script** (`start_streamlit.sh`) to launch the demo web-app for interactive exploration.

---

## 3. Getting Started  
### Prerequisites  
- Python ≥ 3.10  
- Install dependencies (requests, itertools, pyswip, lru_cache, joblib, itertools, anytree, graphviz, json, openai 
from configparser
- (Optional) Prolog/ProbLog/Clingo environment configured if you wish to run logic programs locally.  
- (Optional) Access to large language model API (e.g., OpenAI) or local LLM to generate or evaluate text.
  
---

### Running the Demo  
1. **discourse_parser_app has** an examle of a discourse tree. It uses GPT 5 for discourse parsing. Discourse_parser_endpoint is FastAPI for the parser which returns the same format as the example in the code
2. prolog/**prolog_run_form_app** takes a logic program and executes it.  
3. prolog/**prolog_support_app_w_disc** takes a query and runs it againt both LLM and logic. prolog/prolog_support_app does the same and also uses discourse tree for better rule/constraint attenuation
4. prolog/**problog_app converts** a user request text to probabilistic logic program instead of regular logic program and executes it
5. asp/**argum_text_asp_app** converts a user request text to argumentation program and executes it, relying on answer set programming solverh (clingo)
   

---

## 4. How It Works (Pipeline Overview)  
1. **Discourse Parsing** – identify rhetorical relations (evidence→claim, cause→effect, contrast, etc.).  
2. **Claim Extraction & Mapping** – convert natural-language claims into logic-program predicates/entities.  
3. **Logic Program Runner** – check each claim against the ontology: do they unify? Are there rule violations?  
4. **Rule Attenuation Engine** – simulate weakening of rule conditions to assess plausibility under uncertainty.  
5. **Argumentation Layer** – handle conflicting claims, weigh evidence, determine which conclusions stand.  
6. **Probabilistic Logic Program** – propagate degrees of belief when evidence is incomplete or probabilistic.  
7. **Diagnosis-Making Verifier** – integrate all layers and render final verdict: Accept answer as reliable ✅ / Flag as unreliable ❌ / Suggest repair.  

---

## 5. Use Cases & Applications  
- **Healthcare assistants**: verifying LLM-produced diagnoses or treatment plans before clinician review.  
- **Educational tools**: helping medical students understand why AI suggestions may be unsound by visualizing discourse + logic errors.  
- **Regulatory compliance**: ensuring AI outputs meet explainability and verifiability criteria by grounding in logic and discourse structure.  
- **LLM output hardening**: integrating this pipeline into LLM drifts to detect and correct hallucinations proactively.

---

## 6. Limitations & Future Work  
- Current ontologies and rule-bases are **simplified** and not exhaustive of real-world clinical complexity.  
- Automatic discourse parsing, claim extraction and mapping are **error-prone** – human-in-the-loop remains recommended.  
- Integration with non-English languages, multi-modality (images, labs) and real-time deployment is **ongoing work**.  
- Future work: stronger abductive-logic modules, richer probabilistic modelling, fine-grained argumentation with multi-agent chains.

## 7. Running local discourse parser

Look at /local_discourse_parser

To run in Java, run DiscourseParser.java including the big jar library
it first needs to be compiled:
'javac --release 8 -cp cb_0.11.jar discourse_parser.java'  

Get the big jar with libraries and linguistic models at https://1drv.ms/u/c/02a008d38e197356/EVZzGY7TCKAggAKBAgAAAAABTv2SFOib7yn1xabsjI2gNQ?e=2YAOL9

Then run '& "...\Java\jre1.8.0_471\bin\java.exe" -cp ".;cb_0.11.jar" discourse_parser
'
To run in python, paths needs to be specified:

JAVA_PATH = r"...\Java\jre1.8.0_471\bin\java.exe"

JAR_PATH = os.path.abspath("cb_0.11.jar")

CLASS_PATH = f".;{JAR_PATH}"  # Windows classpath separator is ';'

CLASS_NAME = "DiscourseParser"

Run 'python discourse_wrapper.py'

To keep java in memory, run 'python discourse_jpipe.py'
