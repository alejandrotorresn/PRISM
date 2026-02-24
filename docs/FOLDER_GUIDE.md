# Project Folder Structure Reference

## Current Organization (After Reorganization)

```
Advanced Hybrid Profiler/
│
├── 📁 src/                              # Core Source Code
│   ├── profiler.py                     # ★ Main application (1455 lines)
│   ├── profiler_en.md                  # English documentation excerpt
│   ├── profiler_es.md                  # Spanish documentation excerpt
│   ├── profiler_old.py                 # Legacy/backup version
│   └── __init__.py
│
├── 📁 config/                           # Configuration & Dependencies
│   ├── environment.yml                 # Conda environment specification
│   └── requirements.txt                # Pip dependencies (updated)
│
├── 📁 scripts/                          # Execution Scripts
│   ├── run_experiments.sh              # ★ Main experiment sweep (updated)
│   └── launch_grid5k.sh                # HPC job submission
│
├── 📁 data/                             # Data Storage
│   ├── results/                        # Generated metrics (created at runtime)
│   ├── test-ci/                        # CI test fixtures
│   ├── test-ci-simple/                 # Simplified test data
│   └── test-ci-vit/                    # Vision Transformer test data
│
├── 📁 logs/                             # Execution Logs
│   └── experiments_YYYYMMDD_HHMMSS.txt  # Timestamped experiment logs
│
├── 📁 tests/                            # Unit Tests
│   └── test_timeout_validation.py      # Timeout mechanism tests
│
├── 📁 validation/                       # Validation & Verification
│   ├── validate_code.py                # Quick syntax checks
│   ├── validate_all_models.py          # Model validation
│   ├── validate_zombie_fix.py          # ★ Zombie thread fix validation
│   ├── comprehensive_check.sh          # Full validation suite
│   └── VALIDATION_SUMMARY.sh           # Validation results
│
├── 📁 docs/                             # Documentation
│   ├── README.md                       # Quick start guide
│   ├── documentation.md                # Detailed technical docs
│   ├── FINAL_VALIDATION_REPORT.md      # Test results (60/60 passed)
│   ├── CODE_REVIEW_FINAL_REPORT.md     # Code review findings
│   ├── MODEL_VALIDATION_REPORT.md      # Per-model validation
│   └── ZOMBIE_THREAD_FIX_SUMMARY.md    # ★ Zombie thread issue & fixes
│
├── 📁 .git/                             # Git repository
├── 📁 .github/                          # GitHub configuration
├── 📁 .venv/                            # Python virtual environment
│
├── PROJECT_STRUCTURE.md                 # ★ Detailed folder guide (NEW)
├── QUICK_START.sh                       # ★ Quick reference guide (NEW)
├── .gitignore                           # Git ignore patterns
└── [Other project files]

```

---

## What Each Folder Contains

### 🔧 `src/` - Source Code
**What it is**: Core application logic
**Key file**: `profiler.py` (1455 lines) - the main profiler
**When to edit**: Modify profiling logic, add models, change measurement strategy

### ⚙️ `config/` - Configuration
**What it is**: Environment setup and dependencies
- `environment.yml`: For conda install
- `requirements.txt`: For pip install (updated with complete dependencies)
**When to edit**: Add/remove dependencies, update Python/PyTorch versions

### 🚀 `scripts/` - Execution Scripts
**What it is**: Bash scripts for running experiments
- `run_experiments.sh`: Master script for grid search (updated with zombie thread fix flags)
- `launch_grid5k.sh`: HPC job submission
**When to edit**: Adjust grid search parameters, modify experiment strategy

### 📊 `data/` - Data Storage
**What it is**: Input data and generated results
- `results/`: Where profiler outputs CSV/JSON metrics
- `test-*`: Test fixtures for validation
**When to edit**: Not typically - generated at runtime

### 📝 `logs/` - Execution Logs
**What it is**: Timestamped logs from experiment runs
**When to check**: Debugging failed experiments, monitoring progress

### ✅ `tests/` - Unit Tests
**What it is**: Test code for individual components
**When to edit**: Add tests for new features

### 🔍 `validation/` - Validation Scripts
**What it is**: Integration & system validation
- `validate_zombie_fix.py`: NEW - validates zombie thread fixes
- `comprehensive_check.sh`: Full test suite
**When to run**: Before committing code, after major changes

### 📚 `docs/` - Documentation
**What it is**: Technical guides and reports
- `README.md`: Quick start
- `ZOMBIE_THREAD_FIX_SUMMARY.md`: NEW - explains the zombie thread issue & 3 solutions
- `FINAL_VALIDATION_REPORT.md`: Test results showing 60/60 checks passed
**When to read**: Setup, understanding architecture, troubleshooting

---

## File Organization Rationale

| Original Problem | Solution | New Location | Benefit |
|---|---|---|---|
| Files mixed in root directory | Separate by category | `config/`, `scripts/`, `docs/`, etc. | Clear organization, easier navigation |
| Unclear what each file does | Better naming + folder context | `validation/validate_zombie_fix.py` | Self-documenting |
| Documentation scattered | Centralized in docs/ | `docs/`, `PROJECT_STRUCTURE.md` | Single source of truth |
| Logs fill up root | Moved to logs/ | `logs/experiments_*.txt` | Easier cleanup, git-ignored |
| Test files mixed with code | Separate test folders | `tests/`, `validation/` | Tests ≠ validation |

---

## Quick Navigation

**I need to...**
- ✅ **Profile a model**: Run `python src/profiler.py --model {name}`
- ✅ **Set up environment**: Read `config/environment.yml` or `config/requirements.txt`
- ✅ **Run all experiments**: Execute `bash scripts/run_experiments.sh`
- ✅ **Check output**: Look in `data/results/{model}/`
- ✅ **Monitor progress**: Watch `logs/experiments_*.txt`
- ✅ **Validate installation**: Run `python validation/validate_code.py`
- ✅ **Read documentation**: Check `docs/README.md` or `docs/ZOMBIE_THREAD_FIX_SUMMARY.md`
- ✅ **Understand structure**: See `PROJECT_STRUCTURE.md` (this file)

---

## File Statistics

| Folder | Count | Total Lines | Purpose |
|---|---|---|---|
| `src/` | 4 files | ~2000 | Source code |
| `config/` | 2 files | ~50 | Configuration |
| `scripts/` | 2 files | ~250 | Automation |
| `tests/` | 1 file | ~150 | Unit tests |
| `validation/` | 5 files | ~800 | Integration tests |
| `docs/` | 6 files | ~3000 | Documentation |
| **Total** | **20 files** | **~6250** | **Complete project** |

---

## Key Changes in This Reorganization

### What Was Moved
- ✓ `environment.yml` → `config/`
- ✓ `requirements.txt` → `config/`
- ✓ `run_experiments.sh` → `scripts/`
- ✓ `launch_grid5k.sh` → `scripts/`
- ✓ `*.md` documentation → `docs/`
- ✓ Test scripts → `validation/`
- ✓ `test_timeout_validation.py` → `tests/`

### What Was Updated
- 📝 **requirements.txt**: Added complete dependency descriptions
- 📝 **run_experiments.sh**: Added zombie thread fix configuration (lines 51-52)
- 📝 **src/profiler.py**: No changes (but now with `--skip_cpu` and `--num_threads` args)

### What Was Created
- 📄 **PROJECT_STRUCTURE.md**: Detailed reference (this file)
- 📄 **QUICK_START.sh**: Quick reference guide with examples
- Empty folders: `logs/` (auto-created at runtime)

---

## Git Status After Reorganization

```bash
# Changes to commit:
git add .
git commit -m "refactor: reorganize project structure for clarity

- Move configs to config/ (environment.yml, requirements.txt)
- Move scripts to scripts/ (run_experiments.sh, launch_grid5k.sh)
- Move docs to docs/ (all *.md files)
- Move validation to validation/ (5 validation scripts)
- Move tests to tests/ (test_timeout_validation.py)
- Add PROJECT_STRUCTURE.md for navigation
- Add QUICK_START.sh for quick reference
- Update requirements.txt with complete descriptions
- Update run_experiments.sh with zombie thread fix options
- Delete all desktop.ini files (Windows metadata)
"
```

---

## References

- **Detailed Guide**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) in root
- **Quick Start**: Execute `bash QUICK_START.sh` for interactive guide
- **Current Status**: All files reorganized, all zombie thread fixes integrated

---

*Last Updated*: February 23, 2026
