#!/usr/bin/env python3
"""
Grid5000 Free GPU Node Finder
This script connects to all Grid5000 sites via SSH and uses `oarnodes -J` 
to query the real-time status of nodes. It filters for nodes that:
1. Have a GPU (NVIDIA preferred)
2. Are 'Alive' and not under maintenance
3. Have no active jobs currently running on them.

Note: `oarnodes -J` returns one entry PER OAR RESOURCE (typically per CPU
core), not one entry per physical node. Resources must be grouped by
`network_address` before deciding whether a node is free, otherwise a node
with a job occupying only some of its resources is wrongly reported as free.
"""

import subprocess
import json
import concurrent.futures
import sys

# Official Grid5000 Sites
SITES = [
    "lille", "lyon", "nancy", "nantes", 
    "rennes", "sophia", "grenoble", "luxembourg"
]

def check_site(site):
    """
    Connects to a site frontend via SSH, runs oarnodes in JSON format,
    and parses the output to find free GPU nodes.
    """
    try:
        # We use a ProxyJump (-J) to connect seamlessly from the local machine through the access machine
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-J", "ltorresnino@access.grid5000.fr",
            f"ltorresnino@{site}",
            "oarnodes -J"
        ]
        # Use a timeout so one unresponsive site doesn't hang the script
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        if result.returncode != 0:
            return site, f"Error querying OAR (Exit code {result.returncode})"
            
        data = json.loads(result.stdout)

        # Group per-resource entries by their physical node (network_address),
        # since a single node is represented by many resource_ids in oarnodes -J.
        resources_by_node = {}
        for resource_id, props in data.items():
            node_name = props.get("network_address", "")
            if not node_name:
                continue
            resources_by_node.setdefault(node_name, []).append(props)

        free_nodes = []
        for node_name, resources in resources_by_node.items():
            gpu_model = ""
            compute_cap = ""
            cputype = ""
            cluster = ""
            node_alive = True
            node_busy = False
            node_in_maintenance = False

            for props in resources:
                if props.get("state", "") != "Alive":
                    node_alive = False
                if str(props.get("maintenance", "NO")).upper() == "YES":
                    node_in_maintenance = True
                jobs = props.get("jobs", "")
                if jobs not in ("", None):
                    node_busy = True

                candidate_gpu = props.get("gpu_model", "")
                if candidate_gpu and str(candidate_gpu).lower() not in ["none", "no", "null", ""]:
                    gpu_model = candidate_gpu
                    compute_cap = props.get("gpu_compute_capability_major", "")
                    cluster = props.get("nodeset", "")
                    cputype = props.get("cputype", "")

            # Only report nodes that: have a GPU, are fully Alive, are not
            # under maintenance, and have NO job on ANY of their resources.
            if gpu_model and node_alive and not node_in_maintenance and not node_busy:
                free_nodes.append({
                    "node": node_name,
                    "cluster": cluster,
                    "gpu_model": gpu_model,
                    "compute_cap": str(compute_cap) if compute_cap else "N/A",
                    "cputype": str(cputype) if cputype else "N/A"
                })
                    
        return site, free_nodes
    except subprocess.TimeoutExpired:
        return site, "Timeout while waiting for site response."
    except json.JSONDecodeError:
        return site, "Failed to parse OAR JSON response."
    except Exception as e:
        return site, f"Exception: {str(e)}"

def main():
    print(f"Buscando nodos GPU libres en Grid5000 (Consultando {len(SITES)} sitios en paralelo)...")
    print("-" * 115)
    print(f"{'SITIO':<12} | {'CLUSTER':<12} | {'NODO':<25} | {'MODELO GPU':<18} | {'CAP':<4} | {'CPU TYPE'}")
    print("-" * 115)
    
    total_free = 0
    
    # Use threads to query all sites simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SITES)) as executor:
        futures = {executor.submit(check_site, site): site for site in SITES}
        
        for future in concurrent.futures.as_completed(futures):
            site = futures[future]
            try:
                site, result = future.result()
                if isinstance(result, str):
                    # It's an error message
                    print(f"{site:<12} | {result}")
                elif isinstance(result, list):
                    if len(result) == 0:
                        print(f"{site:<12} | Ningún nodo GPU libre en este momento.")
                    else:
                        seen_nodes = set()
                        for n in result:
                            node_name = n['node']
                            if node_name not in seen_nodes:
                                seen_nodes.add(node_name)
                                print(f"{site:<12} | {n['cluster']:<12} | {node_name:<25} | {n['gpu_model']:<18} | {n['compute_cap']:<4} | {n['cputype']}")
                                total_free += 1
            except Exception as exc:
                print(f"{site:<12} | Fallo general: {exc}")

    print("-" * 95)
    print(f"Búsqueda finalizada. Total de recursos OAR libres con GPU: {total_free}")

if __name__ == "__main__":
    main()
