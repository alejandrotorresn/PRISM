import re
from pathlib import Path

files_to_process = [
    "final_thesis/chapter03_methodological_system_architecture.tex",
    "final_thesis/chapter04_empirical_profiling.tex",
    "final_thesis/chapter05_robust_ilp_formulation.tex"
]

replacements = [
    # Domains & Sets
    (r'\\mathcal\{L\}', r'V'), # Note: \mathcal{L} loss function is not present in ch 3,4,5 as loss
    (r'\\mathcal\{E\}', r'E'),
    (r'\\mathcal\{D\}', r'D'),
    # Node index
    (r'\_n\b', r'_v'),
    (r'\_\{n\}', r'_{v}'),
    (r'\_\{n,d\}', r'_{v,d}'),
    (r'\_\{n,r\}', r'_{v,r}'),
    (r'\_\{n,d,r\}', r'_{v,d,r}'),
    (r'\_\{n,\\mathrm\{GPU\}\}', r'_{v,\\mathrm{GPU}}'),
    (r'\_\{n,\\mathrm\{CPU\}\}', r'_{v,\\mathrm{CPU}}'),
    (r'\^\{d\}\_\{n\}', r'^d_v'),
    (r'\\mathbf\{z\}\_\{n,d,r\}', r'\\mathbf{o}_{v,d,r}'),
    (r'\\forall n \\in', r'\\forall v \\in'),
    (r'\\sum_\{n \\in', r'\\sum_{v \\in'),
    (r'\\sum_\{n\\in', r'\\sum_{v\\in'),
    (r'\{n\\in', r'{v\\in'),
    (r'\\{1,\\dots,L\\}', r'\{1,\dots,|V|\}'),
    (r'\^\(n\)', r'^{(v)}'),
    (r'\^\(n-1\)', r'^{(v-1)}'),
    (r'\\boldsymbol\{\\theta\}_n', r'\\boldsymbol{\\theta}_v'),
    # FLOPs
    (r'f_\{n,r\}', r'\\phi_{v,r}'),
    # Cross phase indicator (Chapter 5)
    (r'v_v \\in \\\{0,1\\\}', r'c_v \\in \\{0,1\\}'), # in case _n already replaced to _v
    (r'\\cdot v_v\}', r'\\cdot c_v\}'), # equation
]

for filepath in files_to_process:
    p = Path(filepath)
    content = p.read_text()
    
    # Custom replacements per chapter
    if "chapter03" in filepath:
        # Assignment vars (remove d subscript and make strictly 1=GPU)
        content = content.replace(r'x^{\mathrm{fwd}}_{n,d}', r'x^{\mathrm{fwd}}_v')
        content = content.replace(r'x^{\mathrm{bwd}}_{n,d}', r'x^{\mathrm{bwd}}_v')
        content = content.replace(r'x^{\mathrm{fwd}}_{v,d}', r'x^{\mathrm{fwd}}_v')
        content = content.replace(r'x^{\mathrm{bwd}}_{v,d}', r'x^{\mathrm{bwd}}_v')
        # We need to manually fix equations 3.5, 3.6, 3.7 later. Let's do the bulk first.

    for old, new in replacements:
        content = re.sub(old, new, content)
    
    p.write_text(content)

print("Bulk regex replacement done.")
