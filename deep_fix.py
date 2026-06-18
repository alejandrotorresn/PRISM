import re

# Chapter 3
path3 = "final_thesis/chapter03_methodological_system_architecture.tex"
with open(path3, "r") as f:
    c3 = f.read()

c3 = c3.replace(r't^{\mathrm{fwd}}_{n,d,r}', r't^{\mathrm{fwd}}_{v,d,r}')
c3 = c3.replace(r'e^{\mathrm{fwd}}_{n,d,r}', r'e^{\mathrm{fwd}}_{v,d,r}')
c3 = c3.replace(r'f_{n,r}', r'\phi_{v,r}')
c3 = c3.replace(r'm^{\mathrm{act}}_{n,r}', r'm^{\mathrm{act}}_{v,r}')
c3 = c3.replace(r't^{\mathrm{bwd}}_{n,d,r}', r't^{\mathrm{bwd}}_{v,d,r}')
c3 = c3.replace(r'e^{\mathrm{bwd}}_{n,d,r}', r'e^{\mathrm{bwd}}_{v,d,r}')

with open(path3, "w") as f:
    f.write(c3)


# Chapter 4
path4 = "final_thesis/chapter04_empirical_profiling.tex"
with open(path4, "r") as f:
    c4 = f.read()

c4 = c4.replace(r'_{n,d,r}', r'_{v,d,r}')
c4 = c4.replace(r't_{n,\mathrm{meas}}^{\mathrm{bwd}}', r't_{v,\mathrm{meas}}^{\mathrm{bwd}}')
c4 = c4.replace(r't_{n,d,r}^{\mathrm{fwd}}', r't_{v,d,r}^{\mathrm{fwd}}')
c4 = c4.replace(r't_{n,d,r}^{\mathrm{bwd}}', r't_{v,d,r}^{\mathrm{bwd}}')
c4 = c4.replace(r't_{n,d,r}', r't_{v,d,r}')
c4 = c4.replace(r'e_{n,d,r}', r'e_{v,d,r}')

with open(path4, "w") as f:
    f.write(c4)


# Chapter 5
path5 = "final_thesis/chapter05_robust_ilp_formulation.tex"
with open(path5, "r") as f:
    c5 = f.read()

c5 = c5.replace(r'x^{\mathrm{fwd}}_{n,\mathrm{GPU}}', r'x^{\mathrm{fwd}}_{v}')
c5 = c5.replace(r'x^{\mathrm{bwd}}_{n,\mathrm{GPU}}', r'x^{\mathrm{bwd}}_{v}')
c5 = c5.replace(r'f^{(n)}', r'\phi_v')
c5 = c5.replace(r'\widetilde{t}^{\mathrm{fwd}}_{n,\mathrm{GPU}}', r'\widetilde{t}^{\mathrm{fwd}}_{v,\mathrm{GPU}}')

with open(path5, "w") as f:
    f.write(c5)

print("Deep fix successful")
