import re

path4 = "final_thesis/chapter04_empirical_profiling.tex"
with open(path4, "r") as f:
    c4 = f.read()

c4 = c4.replace(r'nodo $n$', r'vértice $v$')

with open(path4, "w") as f:
    f.write(c4)
print("nodo n fixed.")
