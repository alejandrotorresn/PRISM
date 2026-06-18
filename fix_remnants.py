import re

# Fix Chapter 3
path3 = "final_thesis/chapter03_methodological_system_architecture.tex"
with open(path3, "r") as f:
    c3 = f.read()

# Fix \mathbf{z} -> \mathbf{o}
c3 = c3.replace(r'\mathbf{z}_{n,d,r}', r'\mathbf{o}_{v,d,r}')
c3 = c3.replace(r'z_{n,d,r}', r'o_{v,d,r}')

# Fix q_{n,d,r} -> q_{v,d,r}
c3 = c3.replace(r'q_{n,d,r}', r'q_{v,d,r}')
c3 = c3.replace(r'\mu_{n,d}^{q}', r'\mu_{v,d}^{q}')
c3 = c3.replace(r'\sigma_{n,d}^{q}', r'\sigma_{v,d}^{q}')
c3 = c3.replace(r'\widetilde{q}_{n,d}', r'\widetilde{q}_{v,d}')

# Fix f_n -> \phi_v
c3 = c3.replace(r'f_{n}', r'\phi_v')

# Fix m_n -> m_v
c3 = c3.replace(r'\widetilde{m}_{n}', r'\widetilde{m}_{v}')

# Fix \widehat{e}_{n,d} -> \widehat{e}_{v,d}
c3 = c3.replace(r'\widehat{e}_{n,d}', r'\widehat{e}_{v,d}')

# Fix \widehat{t}_{n,d} -> \widehat{t}_{v,d}
c3 = c3.replace(r'\widehat{t}_{n,d}', r'\widehat{t}_{v,d}')

# Fix \widehat{m}_{n,d} -> \widehat{m}_{v,d}
c3 = c3.replace(r'\widehat{m}^{\mathrm{act}}_{n,d}', r'\widehat{m}^{\mathrm{act}}_{v,d}')
c3 = c3.replace(r'\widetilde{m}^{\mathrm{act}}_{n}', r'\widetilde{m}^{\mathrm{act}}_{v}')

# Fix \mathcal{E} in equation 252 (line 252) but leave Phase \mathcal{E} alone
c3 = c3.replace(r'\sum_{(u,v)\in\mathcal{E}}', r'\sum_{(u,v)\in E}')

with open(path3, "w") as f:
    f.write(c3)


# Fix Chapter 5
path5 = "final_thesis/chapter05_robust_ilp_formulation.tex"
with open(path5, "r") as f:
    c5 = f.read()

c5 = c5.replace(r'w_{n,\mathrm{GPU}}', r'w_v')
c5 = c5.replace(r'z^{\mathrm{GPU}}', r'z_v') # In the table line 417
c5 = c5.replace(r'c^{\mathrm{fwd}}_{n,d}', r'c^{\mathrm{fwd}}_{v,d}')
c5 = c5.replace(r'c^{\mathrm{bwd}}_{n,d}', r'c^{\mathrm{bwd}}_{v,d}')
c5 = c5.replace(r'\widetilde{t}^{\mathrm{fwd}}_{n,d}', r'\widetilde{t}^{\mathrm{fwd}}_{v,d}')
c5 = c5.replace(r'\widetilde{t}^{\mathrm{bwd}}_{n,d}', r'\widetilde{t}^{\mathrm{bwd}}_{v,d}')
c5 = c5.replace(r'\widetilde{e}^{\mathrm{fwd}}_{n,d}', r'\widetilde{e}^{\mathrm{fwd}}_{v,d}')
c5 = c5.replace(r'\widetilde{e}^{\mathrm{bwd}}_{n,d}', r'\widetilde{e}^{\mathrm{bwd}}_{v,d}')
c5 = c5.replace(r'\omega^{\mathrm{cross}}_{n}', r'\omega^{\mathrm{cross}}_{v}')
c5 = c5.replace(r'\tau^{\mathrm{io}}_{n}', r'\tau^{\mathrm{io}}_{v}')
c5 = c5.replace(r'capa $n$', r'vértice $v$')
c5 = c5.replace(r'la capa $v$', r'el vértice $v$')

# In line 240, there is "la capa $n$" which becomes "la vértice $v$", it should be "el vértice $v$"
c5 = c5.replace(r'la vértice $v$', r'el vértice $v$')

with open(path5, "w") as f:
    f.write(c5)

print("Second pass fixes applied successfully.")
