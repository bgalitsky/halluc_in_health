import os
import jpype
import jpype.imports

# --- Adjust paths for your machine ---
JRE8_JVM = r"C:\Program Files\Java\jre1.8.0_471\bin\server\jvm.dll"  # Java 8 to avoid JAXB issues
CB_JAR = os.path.abspath("cb_0.11.jar")                               # your fat jar
EXTRA_CP = os.path.abspath(".")                                       # where DiscourseParser.class lives
DISABLE_SUTIME = True

_started = False
_DiscourseParser = None

def init_jvm(jvm_path: str = JRE8_JVM, disable_sutime: bool = DISABLE_SUTIME):
    """Start a persistent JVM once, using JPype's classpath support."""
    global _started, _DiscourseParser
    if _started:
        return
    if not os.path.isfile(jvm_path):
        raise RuntimeError(f"JVM not found: {jvm_path}")
    if not os.path.isfile(CB_JAR):
        raise RuntimeError(f"cb_0.11.jar not found at: {CB_JAR}")

    # Start JVM with explicit classpath list (JPype handles its support jar)
    jpype.startJVM(
        jvm_path,
        "-Dfile.encoding=UTF-8",
        classpath=[EXTRA_CP, CB_JAR],
    )
    _started = True

    # Disable SUTime/JollyDay if requested (set after JVM start)
    if disable_sutime:
        from jpype import JClass
        System = JClass("java.lang.System")
        System.setProperty("sutime.binders", "0")
        System.setProperty("ner.applyNumericClassifiers", "false")

    # Load Java class (no package per your setup)
    from jpype import JClass
    _DiscourseParser = JClass("DiscourseParser")

def parse_to_rst(text: str, disable_sutime: bool = DISABLE_SUTIME) -> str:
    if not isinstance(text, str):
        text = str(text)
    if not _started:
        init_jvm(disable_sutime=disable_sutime)
    parser = _DiscourseParser(bool(disable_sutime))
    s = parser.parseToRST(text)
    return (s or "").trim()

def shutdown_jvm():
    global _started, _DiscourseParser
    if _started and jpype.isJVMStarted():
        jpype.shutdownJVM()
    _started = False
    _DiscourseParser = None

if __name__ == "__main__":
    init_jvm()
    print(parse_to_rst("This is a test. However, the result may vary."))
    print(parse_to_rst("A day after the US Senate passed a spending bill to end the longest-ever government shutdown, the budget fight now moves to the House of Representatives. "))
    print(parse_to_rst("The lower chamber of Congress is expected to vote this week on the funding measure. Unlike in the Senate, if House Republicans stay united, they don't need any Democrats to pass the budget. But the margin for error is razor thin." ))
    shutdown_jvm()