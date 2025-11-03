# discourse_jpipe.py  (updated)
import os
import jpype
import jpype.imports
import sys

# --- CONFIG: adjust if needed ---
JRE8_JVM_DLL = r"C:\Program Files\Java\jre1.8.0_471\bin\server\jvm.dll"  # Java 8 JVM (to avoid JAXB issues)
CB_JAR = os.path.abspath("../../../../../discourse_parser/cb_0.11.jar")                                   # your fat jar
EXTRA_CLASSPATH = "."                                                     # where DiscourseParser.class lives
DISABLE_SUTIME = True

def _ensure_jpype_support_jar():
    """Find org.jpype.jar inside the jpype package and expose it via JPYPE_JAR env var."""
    import jpype as _jp
    pkg_dir = os.path.dirname(_jp.__file__)
    candidate = os.path.join(pkg_dir, "org.jpype.jar")
    if os.path.isfile(candidate):
        # Tell JPype exactly where its support jar is.
        os.environ["JPYPE_JAR"] = candidate
        return candidate
    # Some wheels place resources differently; try a few common locations.
    for rel in ("_pyinstaller/org.jpype.jar", "resources/org.jpype.jar"):
        cand = os.path.join(pkg_dir, rel)
        if os.path.isfile(cand):
            os.environ["JPYPE_JAR"] = cand
            return cand
    raise RuntimeError("Cannot locate org.jpype.jar inside jpype package at: " + pkg_dir)

_started = False
_DiscourseParser = None

def init_jvm(jvm_path: str = None, disable_sutime: bool = DISABLE_SUTIME):
    """Start a persistent JVM once."""
    global _started, _DiscouseParser  # typo-safe
    if _started:
        return

    # Ensure JPype can find its own support jar
    support_jar = _ensure_jpype_support_jar()

    jvm = jvm_path or JRE8_JVM_DLL
    if not os.path.isfile(jvm):
        raise RuntimeError(f"JVM not found: {jvm}")

    # Build classpath (Windows uses ';')
    classpath = ";".join(filter(None, [EXTRA_CLASSPATH, CB_JAR, support_jar]))

    jvm_args = [
        "-Dfile.encoding=UTF-8",
        f"-Djava.class.path={classpath}",
    ]
    if disable_sutime:
        jvm_args += [
            "-Dsutime.binders=0",
            "-Dner.applyNumericClassifiers=false",
        ]

    # Start JVM
    jpype.startJVM(jvm, *jvm_args)
    _started = True

    # Load Java class
    _load_discourse_parser()

def _load_discourse_parser():
    """Cache the Java DiscourseParser class reference."""
    global _DiscourseParser  # keep global name unique
    from jpype import JClass
    # DiscourseParser has no package per your setup
    _globals = globals()
    _globals["_DiscourseParser"] = JClass("DiscourseParser")

def parse_to_rst(text: str, disable_sutime: bool = DISABLE_SUTIME) -> str:
    """Parse text and return discourse tree as string."""
    if not isinstance(text, str):
        text = str(text)
    if not _started:
        init_jvm(disable_sutime=disable_sutime)
    parser = _DiscourseParser(bool(disable_sutime))
    res = parser.parseToRST(text)
    return (res or "").strip()

def shutdown_jvm():
    global _started, _DiscourseParser
    if _started and jpype.isJVMStarted():
        jpype.shutdownJVM()
    _started = False
    _DiscourseParser = None

if __name__ == "__main__":
    init_jvm()
    print(parse_to_rst("This is a test. However, the result may vary."))
