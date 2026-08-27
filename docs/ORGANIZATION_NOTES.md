# Repository organization policy

The repository intentionally keeps experiment scripts and result snapshots in stable paths rather than moving them after publication. This preserves links from issues, pull requests, and external citations.

Organization is therefore maintained through indexes and status documents:

- root `README.md` — public overview;
- `docs/README.md` — documentation navigation;
- `docs/RESEARCH_STATUS.md` — claim-status matrix;
- `experiments/README.md` — reproduction-script map;
- `results/README.md` — evidence-file map.

New experiments should update these indexes when they change the public claim boundary. Exploratory or unfinished experiments should remain in a branch/PR until their result status is clear, rather than being presented as a supported finding on `main`.
