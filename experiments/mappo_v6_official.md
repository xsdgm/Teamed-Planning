# Official v6: graph-attention velocity MAPPO

Confirmed as **v6** by the project owner on 2026-10-27. This release is the GAT experiment historically named **v6.2-2**, not the earlier MLP or actor-input-v1 GAT prototypes. Historical filenames and checkpoint metadata are retained for compatibility and reproducibility.

## Canonical configuration

- Training: `src/teamed-planning/config/algorithm/mappo_training_v6.json` (identical settings to `mappo_training_v6_2_2_gat.json`).
- Curriculum: `src/teamed-planning/config/algorithm/mappo_v6_official_distance_curriculum.json`.
- Architecture: `gat_map_v1`; actor input version 2; checkpoint feature version 6. These internal compatibility identifiers are not renamed.
- Existing run: `artifacts/mappo-v6.2-2-gat-stage1-20261026-150400`.
- Exact checkpoint paths and SHA-256 values: [release manifest](mappo_v6_release.json). Checkpoint binaries remain in the ignored experiment directory on bdi; this source release does not upload them to Git.

## Algorithm and information contract

Each shared Actor encodes its own received graph (at most 64 nodes within a +/-96 m window), local 12x12 sensor grid, self state, received mission/neighbor data, and its own CBF margin and chain geometry. A sampled two-dimensional Gaussian latent is radially mapped as `v = 6*a/sqrt(1+||a||^2)` to a desired horizontal velocity. The CBF then produces the executable safe command; vertical velocity is zero.

Training uses a separate centralized Critic and synchronous PPO with 16 episodes per batch, a KL guard, independent PopArt value fitting, and a decaying training-only map-surprise reward. The Actor does not receive centralized Critic state or unobserved global obstacle maps. Global-map reference lengths are used in training reward, budgets and termination; this is privileged training supervision and must be disclosed. The simulator uses ideal local occupancy and relative localization, without a realistic occlusion/noise model. These are not claims of completed hardware deployment validation.

## Stage 1 and checkpoint selection

Stage 1 completed 800 episodes. Best validation score occurred at episode 288 (24/24 map completions); final validation at episode 800 completed 23/24. These are validation results, not the independent paired comparison.

The active task scope is **100-283.808315 m (approximately 100-284 m) only**. Distances beyond this band are excluded from checkpoint selection and future training scope. The historical five-stage curriculum remains for provenance; the command below explicitly stops at stage 1.

The paired comparison uses 12 independent scene seeds (72000-72011) with two action samples per scene, giving 24 episodes per model in the in-scope band. Additional historical evaluation episodes at larger distances are archived and excluded. [In-scope comparison](mappo_v6_paired_comparison_20261027.md): best completed 14/24 with 0/24 CBF terminations; final completed 17/24 with 2/24 CBF terminations. The earlier recommendation based on stage-2 behavior is withdrawn. For the map-completion objective, the provisional same-band continuation candidate is `stage_01_final.pt`; the success difference remains uncertain and the CBF tradeoff is disclosed. No continuation training has started.

## Start a fresh run

```bash
python src/teamed-planning/scripts/simulator/train_distance_curriculum.py \
  --config src/teamed-planning/config/algorithm/mappo_v6_official_distance_curriculum.json \
  --output artifacts/mappo-v6-official-new-run --workers 8 --through-stage 1
```

Use a new output directory. Existing `v6.2-2` checkpoints remain loadable without changing feature versions. See [historical implementation notes](mappo_v6_2_20261026.md) for the input-v2 and batching changes.
