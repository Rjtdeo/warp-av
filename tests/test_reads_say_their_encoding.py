"""Every text file this project reads is UTF-8, and says so.

Python opens text files in the locale's encoding when it is not told otherwise. On the Mac
that is UTF-8 and everything works; on the CARLA laptop it is cp1252, and an em-dash in the
source comes back as three different characters.

Live on 2026-09-15 that failed one test on Windows and nowhere else: test_lane_keeping reads
main.py looking for the sentence "off the line — easing to", and under cp1252 that sentence
does not exist. It had been failing there for some time, quietly, because nobody ran the
suite on that machine often. Sixty-one reads had the same fault waiting.

So: no read of a text file without saying its encoding. Binary reads are exempt -- they are
bytes and have no encoding -- and so are writes, which is a separate question.
"""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOK_IN = ("tests", "src", "tools")
#: a call that opens a file in one of these modes is reading bytes, not text
BINARY = ("rb", "wb", "ab", "r+b", "w+b")


def python_files():
    for top in LOOK_IN:
        for here, _, names in os.walk(os.path.join(ROOT, top)):
            if "__pycache__" in here:
                continue
            for n in names:
                if n.endswith(".py"):
                    yield os.path.join(here, n)


def says_encoding(call):
    return any(k.arg == "encoding" for k in call.keywords)


def mode_of(call):
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
        return call.args[1].value
    for k in call.keywords:
        if k.arg == "mode" and isinstance(k.value, ast.Constant):
            return k.value.value
    return "r"


def offenders():
    bad = []
    for path in python_files():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name == "open" and node.args:
                if str(mode_of(node)) in BINARY or says_encoding(node):
                    continue
                bad.append((os.path.relpath(path, ROOT), node.lineno, "open"))
            elif name in ("read_text", "write_text") and not says_encoding(node):
                bad.append((os.path.relpath(path, ROOT), node.lineno, name))
    return bad


def test_no_text_file_is_read_without_saying_its_encoding():
    bad = offenders()
    shown = "\n".join(f"    {f}:{ln}  {what}(...) with no encoding" for f, ln, what in bad[:20])
    assert not bad, (
        f"{len(bad)} place(s) read or write text without an encoding, so they read differently "
        f"on Windows:\n{shown}\n  add encoding=\"utf-8\"")


def test_the_guard_can_actually_see_one():
    """The guard is only worth having if it catches the thing it is for."""
    tree = ast.parse('open("x.py").read()\nopen("y.py", encoding="utf-8").read()\n'
                     'open("z.bin", "rb").read()\n')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "open"]
    assert len(calls) == 3
    assert not says_encoding(calls[0]) and mode_of(calls[0]) == "r", "this one is the fault"
    assert says_encoding(calls[1]), "this one is fine"
    assert mode_of(calls[2]) in BINARY, "and bytes have no encoding to declare"
