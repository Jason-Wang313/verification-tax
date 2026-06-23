"""Check all \ref and \cite in main.tex for broken references."""
import re

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "main.tex") as f:
    text = f.read()

# Find all labels
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
# Find all refs
refs = re.findall(r'\\(?:ref|cref)\{([^}]+)\}', text)
# Find all cites
cites = re.findall(r'\\(?:cite[tp]?)\{([^}]+)\}', text)

# Check refs
broken_refs = []
for r in refs:
    if r not in labels:
        broken_refs.append(r)

print(f"Labels found: {len(labels)}")
print(f"Refs found: {len(refs)}")
print(f"Broken refs: {len(set(broken_refs))}")
for r in sorted(set(broken_refs)):
    # Find line number
    for i, line in enumerate(text.split('\n'), 1):
        if f'\\ref{{{r}}}' in line or f'\\cref{{{r}}}' in line:
            print(f"  BROKEN: \\ref{{{r}}} on line {i}")
            break

# Check cites against bib
with open(PROJECT_ROOT / "references.bib") as f:
    bib = f.read()
bib_keys = set(re.findall(r'@\w+\{(\w+)', bib))

broken_cites = []
for c in cites:
    for key in c.split(','):
        key = key.strip()
        if key and key not in bib_keys:
            broken_cites.append(key)

print(f"\nCitation keys in bib: {len(bib_keys)}")
print(f"Broken citations: {len(set(broken_cites))}")
for c in sorted(set(broken_cites)):
    print(f"  BROKEN: \\cite{{{c}}}")

# Check for labels defined in appendix that main text references
print(f"\n--- Labels in file ---")
for l in sorted(labels):
    print(f"  {l}")
