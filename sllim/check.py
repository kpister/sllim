import argparse
import ast
import logging
import os
import re
from dataclasses import dataclass, field

from spellchecker import SpellChecker

from . import load_template

logger = logging.getLogger(__name__)

spell = SpellChecker()
spell.word_frequency.load_words(["schema", "bulleted"])


@dataclass
class PromptFile:
    text: str
    filename: str
    typos: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    extra_variables: list[str] = field(default_factory=list)
    missing_variables: list[str] = field(default_factory=list)


@dataclass
class PythonFile:
    filename: str
    text: str
    templates: list[str] = field(default_factory=list)
    uses: dict[str, tuple[str, list[str]]] = field(default_factory=list)


def gather(directory, ignore=None, ext=".txt"):
    if ignore:
        ignore = os.path.abspath(ignore)

    files = os.listdir(directory)
    out_files = []
    for fn in files:
        fn = os.path.join(directory, fn)
        fn = os.path.abspath(fn)
        if ignore and fn.startswith(ignore):
            continue

        if os.path.isdir(fn):
            out_files.extend(gather(fn, ignore, ext))
        elif os.path.isfile(fn) and os.path.splitext(fn)[1] == ext:
            if ext != ".txt" or "prompt" in fn:
                out_files.append(fn)

    return out_files


def check_spelling(pfile: PromptFile):
    results = spell.unknown(spell.split_words(pfile.text))
    pfile.typos = list(
        filter(lambda x: len(x) > 3 and "_" not in x and spell.candidates(x), results)
    )


def load_variables(pfile: PromptFile):
    pfile.variables = set(re.findall(r"\{(\w+)\}", pfile.text))


def check_usage(tfile: PromptFile, pyfiles: list[PythonFile]):
    for pyfile in pyfiles:
        for _, (filename, usages) in pyfile.uses.items():
            if filename not in tfile.filename:
                continue

            for variables in usages:
                for variable in set(variables) - set(tfile.variables):
                    logger.warning(
                        f"{os.path.relpath(pyfile.filename)}: Variable {variable} not defined in {os.path.relpath(tfile.filename)}."
                    )
                    tfile.extra_variables.append(variable)

                for variable in set(tfile.variables) - set(variables):
                    logger.warning(
                        f"{os.path.relpath(pyfile.filename)}: Variable {variable} not used in {os.path.relpath(tfile.filename)}."
                    )
                    tfile.missing_variables.append(variable)


def check_file(filepath, pyfiles: list[PythonFile]):
    prompt = load_template(filepath)

    pfile = PromptFile(text=prompt, filename=filepath)
    check_spelling(pfile)
    load_variables(pfile)
    check_usage(pfile, pyfiles)
    return pfile


def load_python_file(filepath):
    try:
        with open(filepath, "r") as f:
            text = f.read()
    except UnicodeDecodeError:
        logger.warning(f"Unable to read {filepath}.")
        return None

    if "sllim" not in text or "load_template" not in text:
        return None

    track_ids = walk(ast.parse(text))
    python_file = PythonFile(filename=filepath, text=text, uses=track_ids)
    return python_file


def get_arg(node: ast.Call):
    args = node.args or list(map(lambda x: x.value, node.keywords))
    if args:
        if isinstance(args[0], ast.Constant):
            return args[0].value
        elif isinstance(args[0], ast.Name):
            return args[0].id
        elif isinstance(args[0], ast.Call):
            # Use the last constant in the func call as a heuristic
            for arg in args[0].args[::-1]:
                if isinstance(arg, ast.Constant):
                    return arg.value
    return None


def handle_load_template(node: ast.AST):
    if (
        isinstance(node, ast.Call)
        and hasattr(node.func, "id")
        and node.func.id == "load_template"
    ):
        return get_arg(node)

    return None


def handle_use_template(ancestor: ast.AST, node: ast.AST, track_ids):
    if isinstance(node, ast.Name) and node.id in track_ids:
        if (
            isinstance(ancestor, ast.Call)
            and hasattr(ancestor.func, "id")
            and ancestor.func.id == "format"
        ):
            return get_arg(ancestor), ancestor.keywords
    elif isinstance(node, ast.Attribute) and node.attr == "format":
        if isinstance(ancestor, ast.Call):
            return ancestor.func.value.id, ancestor.keywords

    return None, None


def walk(tree: ast.AST):
    track_ids = {}
    for ancestor in ast.walk(tree):
        for child in ast.iter_child_nodes(ancestor):
            if value := handle_load_template(child):
                track_ids[ancestor.targets[0].id] = (value, [])
            else:
                key, keywords = handle_use_template(ancestor, child, track_ids)
                if key and keywords and key in track_ids:
                    track_ids[key][1].append([x.arg for x in keywords])

    return track_ids


def run_checks(text_files, python_files):
    tfiles = []
    pyfiles = []
    for filepath in python_files:
        if pyfile := load_python_file(filepath):
            pyfiles.append(pyfile)

    for filepath in text_files:
        tfiles.append(check_file(filepath, pyfiles))

    spelling_errors = False
    variable_errors = False
    for tfile in tfiles:
        if tfile.typos:
            logger.warning(f"Possible typos in {tfile.filename}: {tfile.typos}")
            spelling_errors = True

        if tfile.extra_variables or tfile.missing_variables:
            variable_errors = True

    if not spelling_errors:
        logger.info("No spelling errors found.")

    if not variable_errors:
        logger.info("No variable errors found.")


def run():
    """Run check command."""
    # All the logic of argparse goes in this function
    parser = argparse.ArgumentParser(description="Basic prompt checking.")
    parser.add_argument(
        "directory",
        type=str,
        help="the name of the directory to search for prompts",
        default=".",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        help="a directory or file to exclude",
        default=None,
        required=False,
    )

    args = parser.parse_args()
    prompt_files = gather(args.directory, args.ignore, ext=".txt")
    prompt_files += gather(args.directory, args.ignore, ext=".prompt")
    python_files = gather(args.directory, args.ignore, ext=".py")
    run_checks(prompt_files, python_files)
