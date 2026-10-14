> Historical execution procedure. Current reports record experiment acceptance; individual earlier attempts may have failed. This is not renewed training authorization. Use [current reports and status](README.md) for actual acceptance and artifact retrieval. Pinned commands/source versions remain preserved below.

# E11 seed42 execution record

Reviewed main317ba670112ffa3123c4bc26baddf9fa1059c60c. Dedicated codex/e11-context64-seed42-20261013
branch; trained clean sourcee7f77205de0c7eabddf6c299029ef5d4b685ffc8; unchanged
scientific hashcdea42adb930904fb0cefcd282d784bb3ac8caa311d3f1e12ef568622f9c12d6.
Only context128→64. Parameters1804800, microbatch4, derived accumulation16,
4096targets/update,64sampledwindows/update. Baseline remains128/8/32windows.
Qualified T4 GPU0/Python3.11.14/torch2.5.1+cu124/CUDA12.4/NumPy1.26.4/freeze.
Canonical raw/tokenizer/allpartition/manifest hashes matched. No test likelihood.

CPU numerical and actual CUDA14-check audits passed; actual Kaggle suite60passed.
Real-data one-update smoke, full sliding validation, checkpoint/ZIP/export/restore,
Agg/generation passed before research reseeding. Exactly5000research updates and
20scheduledfull validations scored55514bytes withcommon256-byte exclusion and
actual64-tokencontext. Best=final5000loss1.49301304/BPB2.15396251.
Training1868.694s,validation267.250s,clone-to-archive2523.689s,peak116.146MiB.
Research savedversion1:rayyannmerchant/nano-e11-pilot-20261013-dafeeadf.

402283332-byte archive SHA256a23789511ce10aa70ac3484ca279f2327eee5a9d1d4db319e4ab2f47df7484ce independently retrieved/CRC/all60filehashes/CPUbestfinalinference passed.
All20optimizer/scheduler counters and private sampler replay passed. All21
snapshotexports were verified natively before publication; independent manifest
and freshCUDA acceptance are recorded separately. Complete100fixed generations,
matchedE01tables and15PNG/15PDFfigures are in the detailed report directory.

Fresh zero-update GPU recovery version1:rayyannmerchant/nano-e11-recovery-20261013-e7000753.
OriginalstateRUNNING/logHTTP500/no receipt delayed delivery. The same job is nowCOMPLETE:exactallstate restoration at5000→5000zeroextra, matchingCPU/CUDA best/final inference. FullE11acceptanceCOMPLETE.
Do not launch duplicate research training. No E12,additionalseed or10k extension.
PR15 is reviewable; do not merge without review. See
../report/pilots/E11_seed42/E11_seed42_analysis.md and operations.md.
