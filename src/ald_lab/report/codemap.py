"""A static map of the ald_lab source: which function calls which, and which module imports which.

Built with ``ast`` from the code that is installed, at report time, so the Code Flow Chart always
matches the code that produced the results. Only links between ald_lab functions are kept; library
calls are dropped. A call through a variable (``cache.target_data(...)``) is resolved when the method
name exists in exactly one ald_lab class.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PYPROJECT = PACKAGE_DIR.parents[1] / "pyproject.toml"
# Used when pyproject.toml is not next to the package (non-editable install).
DEFAULT_ENTRY_POINTS = {
    "ald-build-overview": "cli:build_overview_main", "ald-analyze-params": "cli:analyze_params_main",
    "ald-train": "cli:train_main", "ald-final-test": "cli:final_test_main",
    "ald-predict": "training.predict:main", "ald-export-model": "training.export:main", "ald-report": "cli:report_main",
}
# Method names too generic to resolve through a variable (pandas / dict / polymorphic model methods).
GENERIC = {"values", "items", "keys", "get", "copy", "update", "append", "extend", "add", "pop", "remove", "clear",
           "setdefault", "sort", "insert", "discard", "index", "count", "result", "gather", "key", "runs", "to_frame",
           "to_dict", "from_dict", "fit", "predict", "save", "load", "load_state", "transform", "inverse", "run",
           "render", "summary", "describe", "mean", "sum", "apply", "read", "write", "entries", "close", "join", "split",
           "format", "start", "stop", "forward", "train", "eval", "to", "item", "reset", "__len__", "__post_init__"}
HUB_CALLERS = 6  # a function called from this many places is a shared helper, not a step of a flow
MODULE_CODE = "<실행>"


@dataclass
class Function:
    module: str
    name: str
    lineno: int
    lines: int
    doc: str
    calls: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.module}:{self.name}"


@dataclass
class CodeMap:
    functions: dict[str, Function]
    imports: dict[str, set[str]]  # module → internal modules it imports
    module_lines: dict[str, int]
    module_docs: dict[str, str]
    entry_points: dict[str, str]
    files: dict[str, str] = field(default_factory=dict)  # module → path inside the package ("training/runner.py")

    def module_calls(self) -> dict[str, set[str]]:
        """module → the other modules whose functions it calls."""

        out: dict[str, set[str]] = defaultdict(set)
        for f in self.functions.values():
            for callee in f.calls:
                module = callee.split(":")[0]
                if module != f.module:
                    out[f.module].add(module)
        return out

    def callers(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for f in self.functions.values():
            for callee in f.calls:
                out[callee].add(f.key)
        return out

    def hubs(self) -> set[str]:
        return {k for k, callers in self.callers().items() if len(callers) >= HUB_CALLERS}

    def reach(self, root: str) -> set[str]:
        """Every function ``root`` can end up calling (any depth)."""

        seen, stack = {root}, [root]
        while stack:
            for child in self.functions[stack.pop()].calls:
                if child in self.functions and child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def tree(self, root: str, *, depth: int = 3, limit: int = 40, min_lines: int = 6) -> CallTree:
        """The calls from ``root`` as a tree: each function placed once, under the first caller that reaches it in
        the fewest steps, children in the order the code calls them. Shared helpers and leaf functions shorter
        than ``min_lines`` are left out; other calls between placed functions come back as ``extra`` edges."""

        hubs = self.hubs() - {root}

        def calls(key: str) -> list[str]:
            return [c for c in self.functions[key].calls if c in self.functions and c not in hubs and c != key]

        parent: dict[str, str | None] = {root: None}
        level = {root: 0}
        frontier, skipped = [root], set()
        for d in range(1, depth + 1):
            nxt = []
            for p in frontier:
                for c in calls(p):
                    if c in level:
                        continue
                    leaf = d == depth or not calls(c)
                    if leaf and self.functions[c].lines < min_lines:
                        skipped.add(c)
                        continue
                    if len(level) >= limit:
                        skipped.add(c)
                        continue
                    parent[c], level[c] = p, d
                    nxt.append(c)
            frontier = nxt
        children = {k: [c for c in calls(k) if parent.get(c) == k] for k in level}
        extra = [(k, c) for k in level for c in calls(k) if c in level and parent[c] != k and c != root]
        return CallTree(root, level, children, extra, len(skipped - set(level)))


@dataclass
class CallTree:
    root: str
    level: dict[str, int]
    children: dict[str, list[str]]
    extra: list[tuple[str, str]]  # calls between placed functions that are not tree edges
    hidden: int  # functions reached but left out (small or over the limit)

    def preorder(self) -> list[str]:
        out, stack = [], [self.root]
        while stack:
            key = stack.pop()
            out.append(key)
            stack.extend(reversed(self.children.get(key, [])))
        return out


def _module_name(path: Path, package_dir: Path = PACKAGE_DIR) -> str:
    parts = list(path.relative_to(package_dir).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    if node.level:
        base = module.split(".") if module else []
        if not is_package:
            base = base[:-1]
        if node.level > 1:
            base = base[: len(base) - (node.level - 1)]
        if node.module:
            base += node.module.split(".")
        return ".".join(base)
    if node.module and (node.module == "ald_lab" or node.module.startswith("ald_lab.")):
        return node.module[len("ald_lab"):].lstrip(".")
    return None


def entry_points() -> dict[str, str]:
    if not PYPROJECT.is_file():
        return dict(DEFAULT_ENTRY_POINTS)
    found = {}
    for line in PYPROJECT.read_text(encoding="utf-8").splitlines():
        match = re.match(r'\s*(ald-[\w-]+)\s*=\s*"ald_lab\.([\w.]+):(\w+)"', line)
        if match:
            found[match.group(1)] = f"{match.group(2)}:{match.group(3)}"
    return found or dict(DEFAULT_ENTRY_POINTS)


@dataclass
class _Scope:
    """What the names inside one module refer to."""

    module: str
    own: dict[str, str]  # top-level function name → key
    symbols: dict[str, str] = field(default_factory=dict)  # imported function → key
    class_symbols: dict[str, tuple[str, str]] = field(default_factory=dict)  # imported class → (module, class)
    aliases: dict[str, str] = field(default_factory=dict)  # imported module → module name
    external: set[str] = field(default_factory=set)  # names from other packages (np, pd, torch, …)


@dataclass
class _Index:
    """Every definition in the package, for resolving a name or ``obj.attr`` to a function key."""

    functions: dict[str, Function] = field(default_factory=dict)
    defs: dict[str, dict[str, str]] = field(default_factory=dict)  # module → top-level function name → key
    classes: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))  # module → class names
    properties: set[str] = field(default_factory=set)  # @property methods: reading one is not a call
    methods: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))  # method name → keys

    def method_key(self, module: str, cls: str, attr: str) -> str | None:
        key = f"{module}:{cls}.{attr}"
        return key if key in self.functions and key not in self.properties else None

    def class_of(self, scope: _Scope, name: str) -> tuple[str, str] | None:
        if name in self.classes.get(scope.module, ()):
            return scope.module, name
        return scope.class_symbols.get(name)

    def resolve(self, scope: _Scope, node: ast.AST, cls: str | None, call: bool) -> str | None:
        """The ald_lab function that ``node`` calls (``call``) or passes on as a callback, if any."""

        if isinstance(node, ast.Name):
            if node.id in scope.own or node.id in scope.symbols:
                return scope.own.get(node.id) or scope.symbols[node.id]
            owner = self.class_of(scope, node.id) if call else None
            return self.method_key(*owner, "__init__") if owner else None
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            return None
        base, attr = node.value.id, node.attr
        if base in ("self", "cls"):
            return self.method_key(scope.module, cls, attr) if cls else None
        if base in scope.aliases:  # module.function
            module = scope.aliases[base]
            if attr in self.defs.get(module, {}):
                return self.defs[module][attr]
            return self.method_key(module, attr, "__init__") if call and attr in self.classes.get(module, ()) else None
        owner = self.class_of(scope, base)
        if owner:  # Class.method(...) / Class.classmethod(...)
            return self.method_key(*owner, attr)
        if not call or base in scope.external or attr in GENERIC:
            return None
        candidates = [k for k in self.methods.get(attr, []) if k not in self.properties]
        return candidates[0] if len(candidates) == 1 else None

    def collect(self, scope: _Scope, body: list[ast.stmt], key: str, cls: str | None) -> None:
        calls = self.functions[key].calls
        for stmt in body:
            for node in ast.walk(stmt):
                target = None
                if isinstance(node, ast.Call):
                    target = self.resolve(scope, node.func, cls, True)
                elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(getattr(node, "ctx", None), ast.Load):
                    target = self.resolve(scope, node, cls, False)  # callbacks: pool.map(fn), st.Page(view.render)
                if target and target != key and target not in calls:
                    calls.append(target)


def _defines(index: _Index, name: str, tree: ast.Module, lines: int) -> None:
    """Record the functions, classes and methods of one module (and its top-level code, if it calls anything)."""

    local = index.defs.setdefault(name, {})
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            f = Function(name, node.name, node.lineno, _length(node), _first_line(ast.get_docstring(node)))
            index.functions[f.key] = f
            local[node.name] = f.key
        elif isinstance(node, ast.ClassDef):
            index.classes[name].add(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    f = Function(name, f"{node.name}.{item.name}", item.lineno, _length(item),
                                 _first_line(ast.get_docstring(item) or ast.get_docstring(node)))
                    index.functions[f.key] = f
                    index.methods[item.name].append(f.key)
                    if any(_decorator(d) in ("property", "cached_property") for d in item.decorator_list):
                        index.properties.add(f.key)
    if any(_runs_code(n) for n in tree.body):
        index.functions[f"{name}:{MODULE_CODE}"] = Function(name, MODULE_CODE, 1, lines,
                                                            _first_line(ast.get_docstring(tree)))


def _scope(index: _Index, name: str, tree: ast.Module, is_package: bool, known: set[str],
           imports: dict[str, set[str]]) -> _Scope:
    """Read one module's imports: which local names stand for ald_lab modules, functions and classes."""

    scope = _Scope(name, index.defs.get(name, {}))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            scope.external.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(name, is_package, node)
            if base is None:
                scope.external.update(a.asname or a.name for a in node.names)
                continue
            for alias in node.names:
                local_name = alias.asname or alias.name
                candidate = f"{base}.{alias.name}" if base else alias.name
                if candidate in known:
                    scope.aliases[local_name] = candidate
                    imports[name].add(candidate)
                elif base in known:
                    imports[name].add(base)
                    if alias.name in index.defs.get(base, {}):
                        scope.symbols[local_name] = index.defs[base][alias.name]
                    elif alias.name in index.classes.get(base, ()):
                        scope.class_symbols[local_name] = (base, alias.name)
    return scope


def scan(package_dir: Path = PACKAGE_DIR) -> CodeMap:
    files = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    trees, modules, paths, docs = {}, {}, {}, {}
    for path in files:
        name = _module_name(path, package_dir)
        source = path.read_text(encoding="utf-8")
        trees[name] = (ast.parse(source), path.name == "__init__.py")
        modules[name] = source.count("\n") + 1
        paths[name] = path.relative_to(package_dir).as_posix()
        docs[name] = _first_line(ast.get_docstring(trees[name][0]))
    index = _Index()
    for name, (tree, _) in trees.items():
        _defines(index, name, tree, modules[name])

    imports: dict[str, set[str]] = defaultdict(set)
    for name, (tree, is_package) in trees.items():
        scope = _scope(index, name, tree, is_package, set(trees), imports)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index.collect(scope, node.body, f"{name}:{node.name}", None)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        index.collect(scope, item.body, f"{name}:{node.name}.{item.name}", node.name)
        module_key = f"{name}:{MODULE_CODE}"
        if module_key in index.functions:
            index.collect(scope, [n for n in tree.body if not isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], module_key, None)
            if not index.functions[module_key].calls:  # constants only (re.compile, ...): not a step of any flow
                del index.functions[module_key]
    return CodeMap(index.functions, dict(imports), modules, docs, entry_points(), paths)


def _runs_code(node: ast.stmt) -> bool:
    """A top-level statement that does something when the module runs (a call, an ``if``, a loop …)."""

    if isinstance(node, (ast.Expr, ast.Assign)):
        return isinstance(node.value, ast.Call)
    return not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
                                 ast.AnnAssign))


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n")[0]


def _length(node: ast.AST) -> int:
    return (getattr(node, "end_lineno", None) or node.lineno) - node.lineno + 1


def _decorator(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
