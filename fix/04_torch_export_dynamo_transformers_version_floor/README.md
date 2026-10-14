# Ítem 4 — `torch.export` no podía trazar GPT-2 con datos reales (relleno de atención): piso de versión de `torch`/`transformers`

Relacionado con: ítem 1 (`fix/01_accuracy_convergence/README.md`), descubierto al relanzar la
convergencia de `distilgpt2` tras corregir el ítem 3.

Estado: ✅ **causa raíz encontrada y corregida (piso de versiones actualizado), verificado.**

## El síntoma

Al correr `convergence_validation` para `distilgpt2`, **todos** los modos que necesitan trazar el
grafo (`ilp_plan` en ambos regímenes, `greedy_plan`) fallaban con:

```
status=unsupported
error: Could not trace model 'GPT2LMHeadModel' via symbolic_trace or torch.export; PRISM execution
requires one of these to dispatch operators to CPU/GPU.
```

Esto **no tiene nada que ver con el ILP, con Phase 4, ni con memoria** — el fallo ocurre en el paso
previo de capturar el grafo computacional (necesario para que PRISM sepa qué operaciones existen),
antes de que el ILP decida nada sobre dispositivos.

## Investigación — por qué parecía intermitente/imposible de reproducir

El mensaje de error real queda silenciado por un `except Exception: return None` genérico en
`src/core/decoder_export_backend.py::try_export_decoder_only_trace`. Instrumentando esa función
para no silenciar la excepción, se encontró la causa real:

```
torch._dynamo.exc.Unsupported: Graph break due to unsupported builtin None.finfo.__new__

from user code:
  .../transformers/models/gpt2/modeling_gpt2.py:829, in forward
    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(...)
  .../transformers/modeling_attn_mask_utils.py:158, in _make_causal_mask
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
```

`torch.export` (bajo `torch 2.5.1` + `transformers 4.49.0`, la combinación que fijaba
`config/environment.yml`/`config/requirements.txt`) no puede trazar la llamada a
`torch.finfo(dtype).min` dentro de la construcción de la máscara causal de atención de
`transformers`.

**Por qué era tan difícil de reproducir en pruebas aisladas:** `_prepare_4d_causal_attention_mask_for_sdpa`
solo entra a la rama problemática (`_make_causal_mask`) **cuando el batch tiene relleno real**
(`attention_mask` con ceros, porque alguna línea de texto es más corta que `seq_length`). Un batch
sintético o uno que por casualidad no necesita relleno toma un camino más corto dentro de SDPA que
nunca llega a esa función — y el trazado funciona sin problema. Con datos reales de
`tiny_shakespeare` (líneas de longitud variable), tarde o temprano aparece un batch con relleno real
y el trazado falla. Esto también explica por qué las validaciones anteriores en esta sesión (con
`.venv`, ver abajo) nunca tropezaron con esto — no es que el código sea distinto, es que la versión
de `torch`/`transformers` en `.venv` ya no tiene este problema de compatibilidad con `torch.export`.

## La confirmación

| Entorno | `torch` | `transformers` | ¿Traza `distilgpt2` con datos reales (relleno incluido)? |
|---|---|---|---|
| conda `prism_env` (el fijado en `config/`) | 2.5.1 | 4.49.0 | ❌ No |
| `.venv` (usado en todas las validaciones previas de esta sesión) | 2.10.0+cu128 | 5.2.0 | ✅ Sí |

## Corrección

Se subió el piso de versiones en los 3 archivos que lo fijaban, a la combinación confirmada
funcional (no una adivinada — no se probaron versiones intermedias entre 2.5 y 2.10):

- `config/environment.yml`: `pytorch::pytorch>=2.10,<2.11`, `pytorch::torchvision>=0.25,<0.26`,
  `transformers>=5.0`, `huggingface_hub>=1.0` (transformers≥5.0 ya requiere un cliente de hub de
  esta generación), `pytorch-cuda=12.8`.
- `config/requirements.txt`: `transformers>=5.0.0`, `huggingface_hub>=1.0.0` (torch/torchvision se
  instalan aparte, igual que antes).
- `README.md`: insignia de PyTorch, instrucciones de instalación (`cu121`→`cu128`), y sección
  "System Requirements" con una nota explícita de por qué no se puede bajar de este piso.

## Hallazgo adicional — el canal conda `pytorch` nunca publicó nada después de 2.5.1

Al intentar actualizar `prism_env` en sitio con `conda install -n prism_env -c pytorch -c nvidia
"pytorch>=2.10,<2.11" ...`, conda falló con `PackagesNotFoundInChannelsError: pytorch-cuda=12.8`.
Verificado con `conda search -c pytorch pytorch` y `conda search -c pytorch pytorch-cuda`: el canal
`pytorch` nunca publicó nada más nuevo que `pytorch 2.5.1` / `pytorch-cuda 12.4` — PyTorch dejó de
publicar paquetes conda por completo después de esa versión, a favor de *wheels* de pip. Por eso el
`.venv` usado en todas las validaciones de esta sesión (torch 2.10.0+cu128) tenía que haber sido
instalado por pip, no por conda — no había otra forma de obtener esa versión.

**Corrección en `config/environment.yml`:** se quitaron `pytorch::pytorch`, `pytorch::torchvision`
y `pytorch-cuda=12.8` de la sección `dependencies:` de conda (y los canales `pytorch`/`nvidia`, ya
no necesarios), y se agregaron `torch`/`torchvision` a la sección `pip:` del mismo archivo, con
`--extra-index-url https://download.pytorch.org/whl/cu128`. `conda env create -f
config/environment.yml` ahora instala todo lo demás por conda y torch/torchvision por pip dentro
del mismo entorno, en una sola invocación.

## Resuelto — verificado en el `prism_env` real del usuario

El `prism_env` real terminó en `torch 2.14.1+cu128` / `torchvision 0.29.1+cu128` /
`transformers 5.17.0` (más nuevo que el piso 2.10.0 inicialmente fijado, por haber corrido tanto
`conda env create` con el `environment.yml` corregido como un `pip install --upgrade torch
torchvision` sin tope de versión). Se re-corrió el repro real exacto
(`convergence_validation/run_convergence_study.py --model distilgpt2 --mode ilp_plan --regime B`,
batch 32) contra ese entorno real:

```
[ok] ilp_plan model=distilgpt2 precision=fp32 batch=32 accum=1
  final eval: loss=5.4807, token_accuracy=0.1472
  timing: setup=7.84s, train_loop=3.26s, avg_step=1086.6ms
```

Traza y entrena sin error — confirma que el fix funciona también en 2.14.1, no solo en el 2.10.0
originalmente probado. `config/environment.yml`/`README.md` se actualizaron para pinnear
`torch>=2.14,<2.15` / `torchvision>=0.29,<0.30` (la combinación exacta verificada), en vez del
piso más bajo 2.10.x (nunca verificado end-to-end, solo en el `.venv` de referencia).

`PYTHON_CMD=".venv/bin/python"` ya **no** es necesario en esta máquina — `prism_env` funciona
directamente para `convergence_validation` con `distilgpt2`/`gpt2_small`.

## Nota — instalación de torch/torchvision vía pip, no conda

Al intentar `conda install -n prism_env -c pytorch -c nvidia "pytorch>=2.10,<2.11" ...`, conda
falló con `PackagesNotFoundInChannelsError: pytorch-cuda=12.8`. Verificado con `conda search -c
pytorch pytorch`: el canal `pytorch` nunca publicó nada más nuevo que `pytorch 2.5.1` /
`pytorch-cuda 12.4` — PyTorch dejó de publicar paquetes conda después de esa versión, a favor de
*wheels* de pip. Por eso `config/environment.yml` instala `torch`/`torchvision` vía la sección
`pip:` (con `--extra-index-url https://download.pytorch.org/whl/cu128`), no vía `conda install`.
Para actualizar un `prism_env` ya existente sin recrearlo: `conda activate prism_env && pip install
--upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu128`.
