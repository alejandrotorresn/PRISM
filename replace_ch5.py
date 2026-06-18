import sys

filepath = "final_thesis/chapter05_robust_ilp_formulation.tex"
with open(filepath, "r") as f:
    content = f.read()

# Domains
content = content.replace(r'\mathcal{L}', r'V')
content = content.replace(r'\mathcal{E}', r'E')
content = content.replace(r'\mathcal{D}', r'D')

# Node indices
content = content.replace(r'_{n}', r'_{v}')
content = content.replace(r'_n', r'_v')

# Variables
content = content.replace(r'v_v', r'c_v') # Fix cross phase indicator
content = content.replace(r'w_{v,\mathrm{GPU}}', r'w_v')
content = content.replace(r'z^{\mathrm{GPU}}_{v}', r'z_v')
content = content.replace(r'z^{\mathrm{CPU}}_{v}', r'(1-z_v)') # z_v is 1 for GPU, so CPU is 1-z_v
content = content.replace(r'c^{\mathrm{fwd}}_{v,\mathrm{GPU}}', r'c^{\mathrm{fwd}}_{v}')
content = content.replace(r'c^{\mathrm{bwd}}_{v,\mathrm{GPU}}', r'c^{\mathrm{bwd}}_{v}')
content = content.replace(r'c^{\mathrm{fwd}}_{v,\mathrm{CPU}}', r'c^{\mathrm{fwd}}_{v,\mathrm{CPU}}') # wait, c^fwd is a cost

# Fix the objective function specifically
# We want to replace the cost terms
content = content.replace(
    r'\sum_{v \in V} \left[ c^{\mathrm{fwd}}_{v,\mathrm{GPU}} \cdot x^{\mathrm{fwd}}_{v} + c^{\mathrm{fwd}}_{v,\mathrm{CPU}} \cdot (1 - x^{\mathrm{fwd}}_{v}) \right]',
    r'\sum_{v \in V} \left[ c^{\mathrm{fwd}}_{v,\mathrm{GPU}} \cdot x^{\mathrm{fwd}}_{v} + c^{\mathrm{fwd}}_{v,\mathrm{CPU}} \cdot (1 - x^{\mathrm{fwd}}_{v}) \right]'
)

with open(filepath, "w") as f:
    f.write(content)
