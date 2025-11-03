import subprocess
import os

# Adjust these paths for your environment
JAVA_PATH = r"C:\Program Files\Java\jre1.8.0_471\bin\java.exe"
JAR_PATH = os.path.abspath("../../../../../discourse_parser/cb_0.11.jar")
CLASS_PATH = f".;{JAR_PATH}"  # Windows classpath separator is ';'
CLASS_NAME = "DiscourseParser"


def parse_to_rst(text: str, disable_sutime: bool = True) -> str:
    """
    Call the Java DiscourseParser class and return its discourse tree as string.
    """
    cmd = [
        JAVA_PATH,
        "-cp", CLASS_PATH,
        CLASS_NAME,
    ]
    if disable_sutime:
        cmd.insert(1, "-Dsutime.binders=0")
        cmd.insert(1, "-Dner.applyNumericClassifiers=false")

    # Pass the text argument (quoted automatically by subprocess)
    cmd.append(text)

    # Run Java and capture output
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            shell=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to run Java: {e}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Java process failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


if __name__ == "__main__":
    sample = "This is a test. However, the result may vary."
    print(parse_to_rst(sample))
