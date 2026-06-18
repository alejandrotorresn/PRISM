import sys

filepath = "final_thesis/chapter05_robust_ilp_formulation.tex"
with open(filepath, "r") as f:
    content = f.read()

# Fix \inV and \inD
content = content.replace(r'\inV', r' \in V')
content = content.replace(r'\inD', r' \in D')
content = content.replace(r'\forall n \in', r'\forall v \in')

# Fix n to v in all remaining equations where they were missed
content = content.replace(r'x^{\mathrm{fwd}}_n', r'x^{\mathrm{fwd}}_v')
content = content.replace(r'x^{\mathrm{bwd}}_n', r'x^{\mathrm{bwd}}_v')
content = content.replace(r'x^{\mathrm{fwd}}_{n,d}', r'x^{\mathrm{fwd}}_v')
content = content.replace(r'x^{\mathrm{bwd}}_{n,d}', r'x^{\mathrm{bwd}}_v')

# Fix y variables
content = content.replace(r'y^{\mathrm{fwd}}_{u,v} \in \{0,1\}^{|E|}', r'y^{\mathrm{fwd}}_{u,v} \in \{0,1\}')
content = content.replace(r'\{0,1\}^{|\mathcal{E}|}', r'\{0,1\}^{|E|}')
content = content.replace(r'\{0,1\}^{|\mathcal{L}|}', r'\{0,1\}^{|V|}')
content = content.replace(r'\{0,1\}^{L}', r'\{0,1\}^{|V|}')

# Fix cross-phase indicator c_v (it was v_n)
content = content.replace(r'v_n \in', r'c_v \in')
content = content.replace(r'v_n \geq', r'c_v \geq')
content = content.replace(r'v_n \leq', r'c_v \leq')
content = content.replace(r'v_{v} \in', r'c_v \in')
content = content.replace(r'v_{v} \geq', r'c_v \geq')
content = content.replace(r'v_{v} \leq', r'c_v \leq')
content = content.replace(r'v_{v} = 1', r'c_v = 1')
content = content.replace(r'v_n = 1', r'c_v = 1')
content = content.replace(r'v_n', r'c_v')
content = content.replace(r'v_{v}', r'c_v')

# Fix variables in table
content = content.replace(r'c_v & \{0,1\}^{L}', r'c_v & \{0,1\}^{|V|}')
content = content.replace(r'c_v & \{0,1\}^{|V|}', r'c_v & \{0,1\}^{|V|}')
content = content.replace(r'x^{\mathrm{fwd}}_{n} &', r'x^{\mathrm{fwd}}_{v} &')
content = content.replace(r'x^{\mathrm{bwd}}_{n} &', r'x^{\mathrm{bwd}}_{v} &')
content = content.replace(r'Capa $n$', r'Vértice $v$')
content = content.replace(r'capa $n$', r'vértice $v$')

# Write back
with open(filepath, "w") as f:
    f.write(content)
