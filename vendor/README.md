# Vendored Proton-PottsMPNN

`ProtonPottsMPNN/` is a git submodule of the upstream repo
(github.com/christian-creator/ProtonPottsMPNN), pinned at `09682ab`. The image takes three folders
from it: `foundry/` (the `mpnn` package), `checkpoints/` (the 21 MB v6 design checkpoint) and
`inference/examples/`. Nothing in it is edited; updating means moving the pin.

`ph_worker.py` is ours. It is the only thing a campaign invokes, and it adjusts the engine at
runtime rather than by patching the vendored source.

## What the worker changes, and why it is not visible in the engine source

Reading `foundry/.../inference_engines/potts_mpnn_ph.py` or inspecting `PHDesignCriteria` will not
show any of this. Driving `PottsMPNNPHEngine` directly gets upstream behaviour, unchanged. Only the
worker applies these.

| upstream behaviour | what the worker does | why |
|---|---|---|
| `prepare_potts_input` runs an HBPLUS-dependent protonation labeller | removes that block | HBPLUS is an unredistributable academic binary, and the labels never reach the Potts energy table (the encoder takes no sequence), so `S` falls back to canonical ASP/GLU/HIS, which the checkpoint trained on at `protonation_label_rate=0.5` |
| `_binder_region_masks` calls a residue interfacial if its CA is within `interface_dist=6.0` A of a target CA | recomputes `interface` from heavy-atom contact within `interface_contact_distance` (default 4.5 A), and widens the proxy to 8.0 A as a backstop | a side-chain-mediated interface defeats the CA-CA proxy: on an 86-mer EGFR binder the closest CA-CA approach is 6.03 A, so the mask is empty while 15 residues contact at 2.05 A. Heavy-atom contact is also what BindCraft's own `binder_target_contact_masks` uses |
| `_PHContext.binder_pos_of_res_id` resolves `explicit_centers` against binder positions only | widens the lookup to either chain, preferring the binder on a collision | lets a titratable residue of the fixed target be the centre, with the binder redesigned around it. Nothing downstream needs a pin to be designable |
| `_build_context` hardcodes `designed_chains=[binder]` | replaces it with the campaign's `fixed_residues` | `_annotate_design_scope` applies the `designed_*` fields last, so leaving it would re-open every position the campaign held |
| `forbidden_tokens` permits `ASP-P` / `GLU-P` | adds them by default | a carboxyl pKa of ~3.9-4.3 cannot titrate between pH 6.5 and 7.4 |
| `combined_lambda` defaults to 1.0 | defaults to 0.3 | 1.0 puts no weight on the fold |

## Testing it

Test the integration through the worker, not the engine:

```bash
python ph_worker.py job.json result.json
```

A job is `{"structure": <pdb>, "binder_chain": "B", "fixed_residues": [...], "seeds": [...],
"criteria": {...}, "n_jobs": 1, "seed": 0}`. A reproducer that drives `PottsMPNNPHEngine` directly is
measuring upstream, which is a valid thing to measure and not what a campaign runs.

See `../docs/ph_redesign.md` for the settings and the scientific caveats.
