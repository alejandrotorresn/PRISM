import re
from pathlib import Path

files = [
    Path("final_thesis/chapter03_methodological_system_architecture.tex"),
    Path("final_thesis/chapter04_empirical_profiling.tex"),
    Path("final_thesis/chapter05_robust_ilp_formulation.tex"),
]

for p in files:
    content = p.read_text()
    matches = re.findall(r'\\mathcal\{L\}', content)
    print(f"{p.name}: {len(matches)} \\mathcal{{L}}")
    matches_n = re.findall(r'\_n\b', content)
    print(f"{p.name}: {len(matches_n)} _n")
