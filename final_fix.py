import re

# Fix Chapter 4
path4 = "final_thesis/chapter04_empirical_profiling.tex"
with open(path4, "r") as f:
    c4 = f.read()

c4 = c4.replace(r'm^{\mathrm{total}}_{n,\mathrm{CPU}}', r'm^{\mathrm{total}}_{v,\mathrm{CPU}}')
c4 = c4.replace(r'm^{\mathrm{param}}_{n}', r'm^{\mathrm{param}}_{v}')
c4 = c4.replace(r'm^{\mathrm{act}}_{n,\mathrm{GPU}}', r'm^{\mathrm{act}}_{v,\mathrm{GPU}}')
c4 = c4.replace(r'\eta_{n}', r'\eta_{v}')
c4 = c4.replace(r'\mathrm{TFLOPS}_{n}', r'\mathrm{TFLOPS}_{v}')
c4 = c4.replace(r't^{\mathrm{fwd}}_{n}', r't^{\mathrm{fwd}}_{v}')

with open(path4, "w") as f:
    f.write(c4)

# Fix Chapter 3
path3 = "final_thesis/chapter03_methodological_system_architecture.tex"
with open(path3, "r") as f:
    c3 = f.read()

c3 = c3.replace(r'm^{\mathrm{param}}_{n}', r'm^{\mathrm{param}}_{v}')

with open(path3, "w") as f:
    f.write(c3)

print("Final remnants fixed.")
