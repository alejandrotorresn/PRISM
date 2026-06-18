import re

path5 = "final_thesis/chapter05_robust_ilp_formulation.tex"
with open(path5, "r") as f:
    c5 = f.read()

c5 = c5.replace(r'\sum_{n \in V}', r'\sum_{v \in V}')
c5 = c5.replace(r'c^{\mathrm{fwd}}_{n,\mathrm{GPU}}', r'c^{\mathrm{fwd}}_{v,\mathrm{GPU}}')
c5 = c5.replace(r'c^{\mathrm{fwd}}_{n,\mathrm{CPU}}', r'c^{\mathrm{fwd}}_{v,\mathrm{CPU}}')
c5 = c5.replace(r'c^{\mathrm{bwd}}_{n,\mathrm{GPU}}', r'c^{\mathrm{bwd}}_{v,\mathrm{GPU}}')
c5 = c5.replace(r'c^{\mathrm{bwd}}_{n,\mathrm{CPU}}', r'c^{\mathrm{bwd}}_{v,\mathrm{CPU}}')

with open(path5, "w") as f:
    f.write(c5)
print("n \in V fixed.")
