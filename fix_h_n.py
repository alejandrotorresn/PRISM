import re

path3 = "final_thesis/chapter03_methodological_system_architecture.tex"
with open(path3, "r") as f:
    c3 = f.read()

c3 = c3.replace(r'\mathbf{h}^{(n)}', r'\mathbf{h}^{(v)}')
c3 = c3.replace(r'\mathbf{h}^{(n-1)}', r'\mathbf{h}^{(u)}') # Since u is predecessor of v
c3 = c3.replace(r'f^{(n)}', r'\phi^{(v)}')
c3 = c3.replace(r'\boldsymbol{\delta}^{(n)}', r'\boldsymbol{\delta}^{(v)}')
c3 = c3.replace(r'\boldsymbol{\theta}_n', r'\boldsymbol{\theta}_v')

with open(path3, "w") as f:
    f.write(c3)


path5 = "final_thesis/chapter05_robust_ilp_formulation.tex"
with open(path5, "r") as f:
    c5 = f.read()

c5 = c5.replace(r'\mathbf{h}^{(n)}', r'\mathbf{h}^{(v)}')

with open(path5, "w") as f:
    f.write(c5)

print("Fixed h^(n) and theta_n")
