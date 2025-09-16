import ast
import pathlib
import sys


def scan(root: pathlib.Path):
    for p in root.rglob("*.py"):
        try:
            src = p.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue
        seen = {}
        lines = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                seen[node.name] = seen.get(node.name, 0) + 1
                lines.setdefault(node.name, []).append(node.lineno)
        dups = {k: v for k, v in seen.items() if v > 1}
        if dups:
            print(f"=== {p} ===")
            for name in sorted(dups):
                locs = ", ".join(map(str, sorted(lines[name])))
                print(f"  {name}: {dups[name]} definicje → linie {locs}")
            print()


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    scan(root)
