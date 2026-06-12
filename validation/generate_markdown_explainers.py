import os
from pathlib import Path
import glob

def generate_explainer_for_model(model_dir: Path):
    model = model_dir.name
    md_content = f"# Análisis de Métricas y Rendimiento: {model}\n\n"
    md_content += f"Este documento compila y explica automáticamente todas las visualizaciones generadas para la arquitectura **{model}** durante la campaña doctoral.\n\n"
    
    # 1. Total Nodal Memory Demand
    mem_plots = list(model_dir.rglob("memory_utilization_batch_*.png"))
    if mem_plots:
        md_content += "## 1. Demanda Total de Memoria Nodal\n\n"
        md_content += "Estas figuras muestran el consumo pico (Peak VRAM vs DRAM) de la arquitectura en diferentes tamaños de lote. Cuando el lote excede la capacidad de la GPU, se aplica un mecanismo OOM (Out-of-Memory) y el perfilador utiliza el trazado de la CPU como *fallback* (pues los tensores pesan lo mismo lógicamente). Esto garantiza que el optimizador ILP siempre tenga restricciones válidas.\n\n"
        for p in sorted(mem_plots):
            rel_path = p.relative_to(model_dir)
            md_content += f"![Demanda de Memoria]({rel_path})\n\n"

    # 2. Top 15 Memory Layers
    top_layers = list(model_dir.rglob("top15_memory_batch_*.png"))
    if top_layers:
        md_content += "## 2. Top 15 Capas con Mayor Consumo de Memoria\n\n"
        md_content += "Ilustra las capas específicas que actúan como cuellos de botella de memoria. Las capas densas y los mecanismos de atención masivos suelen acaparar esta métrica. Al igual que en la gráfica anterior, si hay OOM en GPU, se muestran las proyecciones de CPU.\n\n"
        for p in sorted(top_layers):
            rel_path = p.relative_to(model_dir)
            md_content += f"![Top 15 Capas]({rel_path})\n\n"

    # 3. Execution Strategy Comparison (Latency)
    lat_plots = list(model_dir.rglob("strategy_comparison_batch_*.png"))
    if lat_plots:
        md_content += "## 3. Comparativa de Estrategias: Latencia (ms)\n\n"
        md_content += "Muestra el tiempo total de ejecución del pase *Forward+Backward* bajo diferentes estrategias de particionamiento. La heurística *Greedy* satura la VRAM secuencialmente y frecuentemente sufre OOM. *All-GPU* sufre OOM para grandes tamaños de lote. Nuestro particionador **ILP Optimal** siempre encuentra una distribución viable, superando dramáticamente al *All-CPU*.\n\n"
        for p in sorted(lat_plots):
            rel_path = p.relative_to(model_dir)
            md_content += f"![Latencia]({rel_path})\n\n"

    # 4. Energy Comparison
    eng_plots = list(model_dir.rglob("energy_comparison_batch_*.png"))
    if eng_plots:
        md_content += "## 4. Comparativa de Estrategias: Consumo Energético (Joules)\n\n"
        md_content += "Similar a la latencia, pero mide el costo energético. En HPC, el particionamiento ILP suele lograr la menor latencia y la mayor eficiencia energética global, dado que las transferencias PCIe se minimizan óptimamente.\n\n"
        for p in sorted(eng_plots):
            rel_path = p.relative_to(model_dir)
            md_content += f"![Energía]({rel_path})\n\n"

    # 5. Memory Footprint Evolution
    fwd_plot = model_dir / "memory_footprint" / "fwd.png"
    bwd_plot = model_dir / "memory_footprint" / "bwd.png"
    if fwd_plot.exists() and bwd_plot.exists():
        md_content += "## 5. Evolución del Memory Footprint (Forward y Backward)\n\n"
        md_content += "Muestra cómo la memoria VRAM/DRAM se llena a medida que los tensores se propagan. En el *Forward*, las activaciones se acumulan (para usarse en la regla de la cadena). En el *Backward*, se introducen los gradientes, y las activaciones guardadas se van liberando progresivamente (mostradas en el eje secundario).\n\n"
        md_content += f"![Forward Footprint]({fwd_plot.relative_to(model_dir)})\n\n"
        md_content += f"![Backward Footprint]({bwd_plot.relative_to(model_dir)})\n\n"
        
    # 6. Roofline Model
    roofline = model_dir / "roofline" / "roofline.png"
    if roofline.exists():
        md_content += "## 6. Modelo Roofline (Intensidad Aritmética vs Rendimiento)\n\n"
        md_content += "El clásico modelo Roofline de HPC aplicado a las capas individuales. Revela qué capas están limitadas por el rendimiento computacional de la arquitectura (compute-bound) y cuáles están ahogadas por la memoria (memory-bound).\n\n"
        md_content += f"![Roofline]({roofline.relative_to(model_dir)})\n\n"

    out_file = model_dir / "README_Metrics.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Generated {out_file}")

def main():
    plots_dir = Path("reports/zephyr/doctoral_minimal/plots")
    if not plots_dir.exists():
        print("Plots directory not found. Run plot generation first.")
        return
        
    for model_dir in plots_dir.iterdir():
        if model_dir.is_dir() and model_dir.name != "all_models":
            generate_explainer_for_model(model_dir)

if __name__ == "__main__":
    main()
