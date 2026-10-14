# B01 executed commands and evidence boundaries

The research execution source is frozen at
`c2cf1c165d09e0fc8f64a2df714755a4d910d469`. Local reporting and transport fixes
have later commits; they do not change the running notebook or scientific core.
Large outputs use the existing `outputs` junction to D: and dedicated artifact
folders on D:. Credentials and signed URLs are never stored in this record.

## Preparation and actual tests

- Fetched reviewed GitHub main, inspected locked configurations, actual source,
  all fourteen pilot reports and F01 evidence. Reviewed main was
  `d43cfbf7e3af6c982a8bcf1509a5f45b8e17ef4b`.
- CPU architecture audit and real-data uninterrupted4 versus segmented2+2
  continuation; raw six-field checkpoint roundtrip, headless analysis and short
  development generation. CPU validation was explicitly capped512bytes.
- Corrected CPU full suite: 91 passed / one CUDA-only skip /123.63s. Initial
  new tokenizer test failed due to implicit Windows cp1252; explicit UTF8 fixed
  that reader without changing vocabulary or model behavior.
- Qualified development notebook: actual full sliding CUDA continuation and
  92 tests passed /36.88s. Its separately labeled ten-update benchmark measured
  feasibility; projections are estimates, not research runtime measurements.
- Published development ZIP independently downloaded, CRC/safe inventory/all
  59 file hashes and CPU inference verified. First local transport helper failed
  before download due to eager legacy filename fallback; corrected resolver and
  eight regression tests passed without rerunning development training.
- Published development fresh CUDA recovery: exact six-field zero-update
  roundtrip, followed by two smoke-only continuation updates compared with the
  original uninterrupted development state. These are not research updates.
- Local `python -m pytest tests/test_balanced.py tests/test_archive_names.py -q`:
  19 passed /6.33s.
- Local `python -m pytest -q` after report/independent-verifier support:
  99 passed /one CUDA-only skip /109.02s. Fourteen Pyparsing deprecation warnings
  from the local Matplotlib dependency were reported; no test failure.
- `python -m py_compile scripts/balanced/report.py scripts/balanced/inspect.py`
  and `git diff --check` passed.

## GPU notebook commands actually submitted

All launchers invoke the canonical package from a clean pinned checkout. Native
Kaggle Python/PyTorch is left intact; research uses the qualified isolated venv.

```powershell
python -m scripts.balanced.build_notebook --output outputs/B01_preparation/kaggle_development --commit c2cf1c165d09e0fc8f64a2df714755a4d910d469 --mode development
kaggle kernels push -p outputs/B01_preparation/kaggle_development
python -m scripts.balanced.build_notebook --output outputs/B01_preparation/kaggle_development_recovery --commit c2cf1c165d09e0fc8f64a2df714755a4d910d469 --mode development-recovery --source rayyannmerchant/nano-b01-development-20261014-9812dbba
kaggle kernels push -p outputs/B01_preparation/kaggle_development_recovery
python -m scripts.balanced.build_notebook --output outputs/B01_preparation/kaggle_research --commit c2cf1c165d09e0fc8f64a2df714755a4d910d469 --mode research --source rayyannmerchant/nano-b01-development-recovery-20261014-44cabfe4
kaggle kernels push -p outputs/B01_preparation/kaggle_research
```

One research job: `rayyannmerchant/nano-b01-research-20261014-33317a00/1`.
Original engine CLI target-total-step calls are250,500,...,4500, with one run ID
and latest complete checkpoint resumed at each planned boundary. The frozen rule
stopped at4500; no4750/5000 segment was called. Read-only log-stream reconnects
are not GPU submissions or optimizer replay. Final publication/recovery commands
and actual receipts must be added after those checks finish.

## Actual research publication, independent recovery and reporting

Research and fresh recovery saved versions are both COMPLETE. These commands
were executed against the real published outputs, with fresh dedicated D: folders:

```powershell
kaggle kernels status rayyannmerchant/nano-b01-research-20261014-33317a00
kaggle kernels output rayyannmerchant/nano-b01-research-20261014-33317a00/1 -p D:/Nano-Language-Model-artifacts/B01_research_small_20261014_33317a00 --file-pattern '^(artifact_manifest.json|balanced/(completion|stopping_summary|environment|hardware|source_provenance|wall_time|io_timing)\.json|.*\.log)$' -q
python -m scripts.balanced.build_notebook --output outputs/B01_preparation/research_recovery --commit c2cf1c165d09e0fc8f64a2df714755a4d910d469 --mode research-recovery --source rayyannmerchant/nano-b01-research-20261014-33317a00
kaggle kernels push -p outputs/B01_preparation/research_recovery
SDK_PYTHON scripts/pilot/download_kaggle_archive.py --kernel rayyannmerchant/nano-b01-research-20261014-33317a00/1 --manifest D:/Nano-Language-Model-artifacts/B01_research_small_20261014_33317a00/artifact_manifest.json --output D:/Nano-Language-Model-artifacts/B01_research_published_20261014_33317a00 --ranges
PROJECT_PYTHON -m scripts.balanced.inspect --archive D:/Nano-Language-Model-artifacts/B01_research_published_20261014_33317a00/B01_seed42_final.zip --manifest D:/Nano-Language-Model-artifacts/B01_research_small_20261014_33317a00/artifact_manifest.json --output D:/Nano-Language-Model-artifacts/B01_research_published_20261014_33317a00/verified --dataset data/processed/tiny-shakespeare-v3
kaggle kernels output rayyannmerchant/nano-b01-research-recovery-20261014-d2c9b763/1 -p D:/Nano-Language-Model-artifacts/B01_research_CUDA_recovery_20261014_d2c9b763 --file-pattern '^(balanced/(recovery|environment|hardware|generation_reproduction/generation_reproduction)\.json|.*\.log)$' --page-size 200 -q
PROJECT_PYTHON -m scripts.balanced.report --root . --verified D:/Nano-Language-Model-artifacts/B01_research_published_20261014_33317a00/verified --manifest D:/Nano-Language-Model-artifacts/B01_research_small_20261014_33317a00/artifact_manifest.json --cuda-receipt D:/Nano-Language-Model-artifacts/B01_research_CUDA_recovery_20261014_d2c9b763/balanced/recovery.json --output D:/Nano-Language-Model-artifacts/B01_report_staging_20261014_33317a00 --dataset data/processed/tiny-shakespeare-v3 --kernel rayyannmerchant/nano-b01-research-20261014-33317a00
```

Here SDK_PYTHON denotes the actual
`C:/Users/RayyanMerchant.Inter/AppData/Roaming/uv/tools/kaggle/Scripts/python.exe`,
and PROJECT_PYTHON is `.venv/Scripts/python.exe`; `kaggle` is
`C:/Users/RayyanMerchant.Inter/.local/bin/kaggle.exe`. Scripts were invoked with
these actual executables, not a different native Kaggle research dependency profile.

Download396.989s verified full604,825,667bytes/SHA256; CPU audit60.398s passed
all96 hashes/counters/complete4,500-step journal/18 validations/sampler replay.
Fresh CUDA raw all-six-field4500→4500/37 snapshots/100 reproduced strings passed
with zero optimizer updates. A separate local read-only assertion compared all
100 original/recovered256-token ID arrays exactly and CPU64-token ROMEO greedy
probes against original CUDA256-token prefixes; both policies pass.

All56 small staging report/receipt/figure files were copied to report/balanced
with SHA256 equality and no earlier file overwritten. Full-grid qualitative
inspection and the model-selection narrative were written after reading all100
original strings; numerical tables and plots remain generated by the shared
analysis/report utilities. Original measurements and historical reports were
not rewritten. Native figures and three comparison figures were visually checked.
