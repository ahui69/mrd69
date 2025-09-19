import ast, os, sys, importlib.util

# stdlib (Py3.10+), fallback gdyby brakowało:
try:
    STDLIB = set(sys.stdlib_module_names)  # Python 3.10+
except Exception:
    STDLIB = {
        "abc","argparse","asyncio","base64","collections","contextlib","copy","csv","datetime","enum",
        "functools","glob","hashlib","heapq","html","http","io","itertools","json","logging","math",
        "mimetypes","os","pathlib","pickle","platform","queue","random","re","shutil","socket","sqlite3",
        "statistics","string","subprocess","sys","tempfile","textwrap","threading","time","typing","uuid",
        "urllib","warnings","xml","zipfile","dataclasses"
    }

# mapowanie aliasów  nazwy paczek na PyPI
NAME_MAP = {
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "bcrypt": "bcrypt",
    "jinja2": "Jinja2",
    "regex": "regex",
    "dotenv": "python-dotenv",
    "uvicorn": "uvicorn",
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "openai": "openai",
    "tqdm": "tqdm",
    "sqlalchemy": "SQLAlchemy",
    "psycopg2": "psycopg2-binary",
    "lxml": "lxml",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "requests": "requests",
    "beautifulsoup4": "beautifulsoup4",
    "Pillow": "Pillow",
    "sentry_sdk": "sentry-sdk",
    "orjson": "orjson",
    "ujson": "ujson",
    "pyarrow": "pyarrow",
    "tenacity": "tenacity",
    "rich": "rich",
    "coloredlogs": "coloredlogs",
    "loguru": "loguru",
}

# katalogi do skanu (repo + typowe podkatalogi)
SCAN_DIRS = ["."]

def is_local(module_name: str) -> bool:
    """Odrzuć importy, które są lokalnymi modułami projektu."""
    rel = module_name.replace(".", os.sep)
    for base in SCAN_DIRS:
        for ext in (".py", os.sep+"__init__.py"):
            if os.path.exists(os.path.join(base, rel + ext)):
                return True
    return False

def collect_imports():
    found = set()
    for base in SCAN_DIRS:
        for root, _, files in os.walk(base):
            # pomijamy venvy i cache
            if any(part.startswith(".venv") or part == "__pycache__" for part in root.split(os.sep)):
                continue
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                            tree = ast.parse(fh.read(), filename=path)
                    except Exception:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                top = (n.name.split(".")[0]).strip()
                                if top: found.add(top)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                top = (node.module.split(".")[0]).strip()
                                if top: found.add(top)
    return found

def to_pypi_name(mod: str) -> str:
    if mod in NAME_MAP:
        return NAME_MAP[mod]
    return mod

def filter_external(mods):
    external = set()
    for m in mods:
        # stdlib?
        if m in STDLIB:
            continue
        # lokalny moduł?
        if is_local(m):
            continue
        # ignoruj puste/techniczne
        if not m or m == "_":
            continue
        external.add(to_pypi_name(m))
    return sorted(external, key=str.lower)

def main():
    mods = collect_imports()
    reqs = filter_external(mods)
    # heurystyczny zestaw pewniaków na podstawie importów w seed_memory i podobnych
    SUGGEST = []
    # unikaj duplikatów
    final = []
    for r in reqs + SUGGEST:
        if r not in final:
            final.append(r)
    if not final:
        print("Nie wykryto zewnętrznych zależności. (Albo wszystko jest stdlib / lokalne.)")
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for r in final:
            f.write(r + "\n")
    print("Wygenerowano requirements.txt z", len(final), "pozycjami.")
    for r in final:
        print(" -", r)

if __name__ == "__main__":
    main()

