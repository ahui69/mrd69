import ast
import pathlib
import sys


def dedupe_defs(path: pathlib.Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"[ERROR] {path}: {e}")
        return  # idź dalej z innymi plikami

    defs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(
            n, "end_lineno"
        ):
            kind = "class" if isinstance(n, ast.ClassDef) else "def"
            defs.append((n.name, kind, n.lineno, n.end_lineno))

    by = {}
    for name, kind, s, e in defs:
        by.setdefault((name, kind), []).append((s, e))

    to_remove = []
    for key, ranges in by.items():
        if len(ranges) > 1:
            ranges.sort(key=lambda x: x[0])
            to_remove += [(s, e, key) for (s, e) in ranges[:-1]]

    if not to_remove:
        return

    lines = text.splitlines(keepends=True)
    to_remove.sort(key=lambda x: x[0], reverse=True)
    for s, e, _ in to_remove:
        del lines[s - 1 : e]

    path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
    path.write_text("".join(lines), encoding="utf-8")
    print(f"[FIX] {path}  (-{len(to_remove)} duplikatów)")


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    dedupe_defs(p)
