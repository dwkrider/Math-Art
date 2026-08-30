"""Lint the LaTeX in the generator doc pages.

GitHub renders `$...$` and `$$...$$` with MathJax.  Backslash *runs*
(`\\\\`) and command names (`\\alpha`) pass through unchanged, but the
Markdown layer still unescapes single-character escapes like `\\_` before
MathJax runs -- so an escaped underscore inside `\\text{}` does NOT
survive (see check 5).  Mistakes that are easy to make and invisible in
the source:

  * `\\\\\\\\` (four backslashes) where `\\\\` (two) is meant.  MathJax
    reads two row separators and inserts a blank row in the matrix.
  * `$10\\,$cm` -- a thin space left inside the math and the unit outside,
    which renders as "10" then an unspaced "cm" in body text.

Also checks that `$` and `$$` are balanced, since an unclosed one turns
the rest of a paragraph into maths.

Run:  python tools/check_docs_math.py
Exits non-zero if anything is found, so it can gate a docs pass.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(HERE, "docs", "generators")


def strip_display(text):
    """Remove $$...$$ blocks, returning the rest (for inline checks)."""
    return re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)


def math_only(text):
    """Blank out everything that is NOT inside math, preserving length and
    newlines (so line numbers still line up).  The math-specific checks run
    against this so a literal `|` escaped for a Markdown table cell (\\|) in
    a generated Options row is not mistaken for a norm bar in math."""
    mask = [" "] * len(text)
    for pat, flags in ((r"\$\$.*?\$\$", re.DOTALL),
                       (r"(?<!\$)\$[^$\n]+\$(?!\$)", 0)):
        for m in re.finditer(pat, text, flags):
            mask[m.start():m.end()] = text[m.start():m.end()]
    for i, c in enumerate(text):
        if c == "\n":
            mask[i] = "\n"
    return "".join(mask)


def check(path):
    name = os.path.basename(path)
    text = open(path, encoding="utf-8").read()
    mtext = math_only(text)   # math-only view for the math-specific checks
    problems = []

    # 1. over-escaped backslash runs.  Three or more in a row is always
    #    wrong: LaTeX uses one (\alpha) or two (row separator).
    for m in re.finditer(r"\\{3,}", text):
        line = text.count("\n", 0, m.start()) + 1
        problems.append((line, "%d consecutive backslashes (LaTeX wants 1 or 2)"
                         % len(m.group(0))))

    # 2. a CLOSING `$` glued to a word character -- `$10\,$cm`.  Match
    #    whole inline spans rather than bare `$`, or the opening
    #    delimiter of every ordinary `$x$ text` trips the check.
    for m in re.finditer(r"(?<!\$)\$([^$\n]+)\$(?!\$)", strip_display(text)):
        after = m.string[m.end():m.end() + 1]
        if after and (after.isalnum() or after == "\\"):
            line = text.count("\n", 0, m.start()) + 1
            problems.append((line, "math closes straight into text: $%s$%s"
                             % (m.group(1)[:24], after)))

    # 3. balance.  Count `$$` first, then the single `$` that remain.
    if text.count("$$") % 2:
        problems.append((0, "odd number of $$ delimiters"))
    inline = strip_display(text).count("$")
    if inline % 2:
        problems.append((0, "odd number of inline $ delimiters (%d)" % inline))

    # 4. a thin space immediately before a closing delimiter is a sign of
    #    the unit-outside-the-math mistake even when a space follows.
    for m in re.finditer(r"\\,\$", text):
        line = text.count("\n", 0, m.start()) + 1
        problems.append((line, "thin space `\\,` right before closing $"))

    # 5. an underscore inside \text{} (or \textbf/\texttt/... ).  GitHub's
    #    Markdown consumes the backslash of an escaped `\_` before MathJax
    #    runs, so BOTH `\text{a_b}` and `\text{a\_b}` reach MathJax as a raw
    #    `_` in text mode -> "'_' allowed only in math mode".  Keep code
    #    identifiers out of math: use a symbol and name the identifier in
    #    prose (backticks).  The pattern matches the escaped form too,
    #    because `\_` still contains a literal `_`.
    for m in re.finditer(r"\\text[a-z]*\{[^}]*_[^}]*\}", mtext):
        line = text.count("\n", 0, m.start()) + 1
        problems.append((line, "underscore inside `%s` -- breaks on GitHub; "
                         "use a symbol in math, name the identifier in prose"
                         % m.group(0)[:36]))

    # 6. macros GitHub's MathJax allowlist rejects outright ("The following
    #    macros are not allowed: ...").  \operatorname is the common trap --
    #    use \mathrm{...} instead; definition macros are blocked for safety.
    for m in re.finditer(r"\\(operatorname|def|newcommand|renewcommand|gdef"
                         r"|let|input|include|require|catcode"
                         r"|DeclareMathOperator)\b", mtext):
        line = text.count("\n", 0, m.start()) + 1
        problems.append((line, "GitHub-disallowed macro `\\%s` "
                         "(use \\mathrm{} for operator names)" % m.group(1)))

    # 7. backslash-escaped braces/pipe in math.  GitHub's Markdown unescapes
    #    \{ \} \| to { } | before MathJax, so \left\{ becomes the invalid
    #    \left{ ("Missing delimiter for \left") and bare \{...\} lose their
    #    braces.  Use the command forms \lbrace \rbrace \Vert, which survive.
    for m in re.finditer(r"\\[{}|]", mtext):
        line = text.count("\n", 0, m.start()) + 1
        sym = {"{": "\\lbrace", "}": "\\rbrace", "|": "\\Vert"}[m.group(0)[1]]
        problems.append((line, "escaped `%s` in math -- GitHub eats the "
                         "backslash; use `%s`" % (m.group(0), sym)))

    return name, problems


def main():
    total = 0
    for f in sorted(os.listdir(GEN)):
        if not f.endswith(".md"):
            continue
        name, problems = check(os.path.join(GEN, f))
        for line, msg in problems:
            where = "%s:%d" % (name, line) if line else name
            # the pages contain plenty of non-cp1252 mathematics; do not
            # let the console encoding turn a lint run into a crash
            out = ("  %-32s %s" % (where, msg))
            sys.stdout.write(out.encode(sys.stdout.encoding or "utf-8",
                                        "replace")
                             .decode(sys.stdout.encoding or "utf-8") + "\n")
            total += 1
    print("\n%d problem(s) in %d pages"
          % (total, len([f for f in os.listdir(GEN) if f.endswith(".md")])))
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
