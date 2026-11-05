# pH-switch redesign

An optional stage that turns the sequence step into a pH-conditioned one. ProteinMPNN draws the
candidates as it always does; Proton-PottsMPNN then pins a protonated centre and redesigns its
neighbourhood so the binder's energy differs between the centre's protonated and deprotonated states.
That difference is the switch: protonation state follows pH, so a design whose interface is favourable
only while the centre carries a proton binds at low pH and releases at neutral.

It is off unless `ph_redesign` is set. With it off, nothing in this document is imported and a campaign
is identical to stock BindCraft2.

## Running it

```bash
docker build -t bindcraft2:ph -f containers/Dockerfile.ph .
docker run --gpus all -v "$PWD:/work" bindcraft2:ph design settings.json
```

The smallest settings change that enables the stage:

```json
{
  "ph_redesign": true
}
```

Defaults then apply: one `HIS-P` centre, placed at the interface by `scan_potts`, `combined_lambda`
0.3, up to 20 mutations within 16 coupled neighbours of the centre.

`combined_lambda` defaults to 0.3 rather than 1.0 because the objective is
`(1-lambda)*z(stability) + lambda*z(selective)`, so 1.0 puts no weight at all on the fold. On an
86-mer EGFR design, 1.0 rewrote 12 of 86 positions and replaced a helix with an acidic block where
0.3 moved 5 and left the sequence recognisable.

## Settings

| key | default | meaning |
|---|---|---|
| `ph_redesign` | `false` | enable the stage |
| `ph_criteria` | `{}` | passed through to `PHDesignCriteria`; see below |
| `ph_workers` | `8` | CPU fork-pool width for the λ sweep |
| `ph_timeout` | `1800` | seconds before one backbone's pass is abandoned |
| `ph_strict` | `false` | raise on worker failure instead of keeping the ProteinMPNN sequences |
| `ph_redesign_interface` | `false` | let the pH pass redesign the paratope even when ProteinMPNN holds it |
| `ph_rank` | `combined` | which design to keep per seed: `combined`, `selective` or `stability` |

Leave `ph_strict` off for a long campaign so one bad backbone cannot end it, and turn it on while
setting a new target up, so a misconfiguration is loud rather than silently a no-op.

### `ph_criteria`

Any field of `PHDesignCriteria` plus `lambdas`, which sweeps `combined_lambda` and is the usual way to
ask for several designs per seed. The ones that matter:

```json
{
  "ph_redesign": true,
  "ph_criteria": {
    "lambdas": [0.0, 0.33, 0.67, 1.0],
    "center_types": ["HIS-P"],
    "dep_map": {"HIS-P": ["HIS-S"]},
    "placement_region": ["interface"],
    "placement_by": "scan_potts",
    "neighbour_k": 16,
    "max_mutations": 20,
    "block_size": 3
  }
}
```

- `center_types` is an exact composition to place. `["HIS-P", "HIS-P"]` places two histidine centres,
  which is how you multiply a marginal switch into a usable one.
- `lambdas` trades stability against selectivity: 0 is pure stability, 1 pure selectivity. Each value
  produces designs for every seed, and the best per seed by selective energy is the one kept.
- `block_size` costs `V**block_size` per block, with `V` = 30. Three is the shipped default; four is
  about 30x the work for a marginally better optimum.
- `interface_contact_distance` (default 4.5 A) sets what `placement_region: ["interface"]` means. It
  is a heavy-atom contact test, not the engine's CA-CA proxy: see the note below.
- `allow_target_centers` (default true) permits `explicit_centers` to name a residue of the target.

### `ph_rank` and lambda sweeps

Ranking on selective energy alone makes a sweep self-defeating, because the `lambda=1` member wins by
construction and the sweep collapses onto its least stable arm. The default `combined` scores each
design on the objective its own criteria optimised, with the z-scores taken across the batch so the
two energies are comparable. Use `selective` to recover the older behaviour, or `stability` to keep
the soundest fold that still carries a centre.

### What "interface" means here

`placement_region: ["interface"]` uses heavy-atom contact within `interface_contact_distance`, which
is the same test BindCraft's own `binder_target_contact_masks` applies, so the two layers agree.

The engine's own definition is a CA-CA proxy at 6.0 A, which a side-chain-mediated interface defeats:
on an 86-mer EGFR binder the closest CA-CA approach is 6.03 A, so that mask is **empty** while 15
residues are in heavy-atom contact at 2.05 A. Placement then has nowhere to put a centre and the
stage returns nothing. Raising the CA-CA cutoff would hide this at a value specific to one fold and
one target, so the metric is replaced instead. The worker logs both counts per backbone.

### Choosing the centre chemistry

The pH at which a switch flips is set by the pKa of the residue pinned, not by any number here. There
is no pH parameter in the model.

| centre | side-chain pKa | flips around |
|---|---|---|
| `HIS-P` | ~6.0-6.5 | pH 6 to 7 |
| `GLU-P` | ~4.3 | pH 4 to 4.5 |
| `ASP-P` | ~3.9 | pH 4 |

For a 6.5-versus-7.4 window, histidine is the chemistry; aspartate and glutamate are two to three pH
units off unless deeply buried.

`forbidden_tokens` therefore defaults to the engine's ambiguous tokens **plus `ASP-P` and `GLU-P`**.
Upstream permits those, and the infill will then quietly introduce protonated acid centres nobody
asked for: on the shipped PD-L1 example, asking for one `HIS-P` produced six centres, five of them
`ASP-P`, and about 36% of the apparent selective gap came from carboxyls that are not protonated at
pH 6.5 at all. Set `"forbid_extra_protonated": false` in `ph_criteria` to restore upstream behaviour,
and only do that if your window is acidic enough to populate them.

### Pinning a centre on the target

The default places the centre on the binder. If the switch should instead come from a titratable
residue of the *target*, pin it explicitly and the binder's neighbourhood is redesigned around it:

```json
"ph_criteria": {
  "explicit_centers": [{"res_id": 409, "protonation_type": "HIS-P"}],
  "dep_map": {"HIS-P": ["HIS-S"]}
}
```

The centre needs no design permission: only coupled binder positions are ever redesignable, so a
residue of the fixed target is a valid centre and the binder is rebuilt around it. Upstream resolves
explicit centres through a binder-only lookup and raises for a target residue; this image widens the
lookup to either chain, preferring the binder when a `res_id` appears on both, and logs whenever a
centre resolves off the binder. Set `allow_target_centers: false` to restore upstream behaviour.

`res_id` is read from the structure as the campaign writes it, so it must match the numbering of the
target you are running against, which for a cropped target is the crop's numbering rather than the
full-length one. Check it before trusting a run.

## What the stage respects

Every position the campaign holds is held here too: the whole target, a kept interface when
`redesign_interface` is false, a fold-conditioning framework, a multi-domain linker, and an induced-fit
mobile set. They are passed as fixed residues derived from the same `ResidueFlags.DESIGN` the
ProteinMPNN stage uses, so the two stages cannot disagree.

Residues the campaign excluded outright are excluded here too. An `aa_bias` weight of zero or less
becomes an omission upstream (`settings.resolve_omitted_amino_acids`), and those are forwarded as
forbidden canonical tokens, so the pH pass cannot reintroduce a residue the campaign forbids. Soft
biases are not forwarded: the pH pass optimises the Potts objective and can shift composition, which
is what the campaign's own composition filters are there to catch.

`redesign_interface` defaults to false, which hands the paratope over held, and that is exactly what a
pH centre needs to perturb. Three options, in increasing bluntness: pin the centre on the target (the
coupling graph reaches past the contact shell, so second-shell positions stay designable); set
`ph_redesign_interface: true` to let the pH pass alone touch the paratope while ProteinMPNN keeps
preserving it; or set `redesign_interface: true` and change the ProteinMPNN stage for the whole
campaign. The middle option is usually what is wanted.

## When the stage does nothing

A run that pins no centre is a failure, not a success. Placement can legitimately find nowhere to put
one, and the worker then returns an empty design list with no error set; treated as success that would
quietly run the campaign as stock BindCraft2 with an empty `ph_redesign.jsonl` as the only evidence.
The stage counts it as a failure, says why, and `ph_strict` raises on it. After five such batches in a
row it says plainly that the stage is a no-op as configured.

A complex with no designable binder position at all is refused before the worker runs, because
featurising one has been observed to hang the engine rather than return empty, which `ph_timeout`
would otherwise absorb once per trajectory.

Holding the paratope does not by itself empty the neighbourhood when the centre is on the target: the
Potts coupling graph is a 48-nearest-neighbour graph and reaches well past the 4 A contact shell, so
second-shell binder positions stay designable. That makes a target-side centre workable even at
`redesign_interface: false`, with the redesign pushed one shell out.

## Output

One JSON line per kept candidate at `<project folder>/ph_redesign.jsonl`:

```
{"candidate": 1, "seed_sequence": "...", "mutations_vs_seed": 11,
 "design_id": "HIS-P_block_descent_b3_l1_seed0_res28_s0", "sequence": "...",
 "potts_energy": -51636.1, "selective_energy": -6.65, "global_protonation_dH": null,
 "combined_lambda": 1.0, "center_res_ids": [28], "center_protonation_types": ["HIS-P"],
 "n_neighbours": 11}
```

`selective_energy` is the centre's protonated-minus-deprotonated gap, and lower means the protonated
state is more preferred, so it is the switch magnitude. `potts_energy` is the whole-system Hamiltonian,
lower being stabler. Both are in arbitrary model units.

## Limits worth knowing before you trust a number

- **The energies are uncalibrated.** `selective_energy` ranks candidates; it is not a pKa, a ΔΔG or an
  affinity ratio. Upstream validates it only by rank correlation against experimental pKa. Screen the
  survivors with a real pKa predictor before believing any particular design switches in your window.
- **AlphaFold cannot see protonation.** It folds the 20-letter canonical projection, so the campaign's
  own filters say nothing about the switch. They still check the design folds and binds, which is what
  you want from them, but objective evidence for the switch has to come from elsewhere.
- **No tied decoding.** Block descent has no equivalent of the tied ProteinMPNN path, so for symmetric
  oligomers and tied multitarget the pass runs on one state's backbone and the sequence is shared, which
  is what the stage does downstream anyway.
- **Low-complexity drift.** Block descent tends toward aspartate clusters even with `ASP-P` forbidden.
  `repetitive_window_weight` with `repetitive_window_parents: ["ASP", "GLU"]` disperses them at some
  cost in selectivity. The campaign's own composition filters are the backstop.

## Iterating

The package is installed editable, so host edits apply without a rebuild:

```bash
docker run --gpus all \
  -v ~/BindCraft2:/opt/bindcraft \
  -v ~/BindCraft2/vendor/ph_worker.py:/opt/protonpotts/ph_worker.py:ro \
  -v "$PWD:/work" bindcraft2:ph design settings.json
```

## How it is put together

The engine runs in `/opt/protonpotts/venv`, a second interpreter, because `atomworks` pins
`biotite==1.4.0` while this package needs 1.7.x; they cannot share one `site-packages`. torch is the
CPU build: the engine's fork pool is gated on `device == "cpu"` so a GPU is unusable to it anyway, and
a CUDA torch would add a second set of `nvidia/*/lib` directories to
`/etc/ld.so.conf.d/bindcraft-cuda.conf`, where `ldconfig` would choose one cuDNN globally and could
break jax. The hop costs nothing against a featurisation of roughly 18 seconds per backbone, after
which designs are close to free: 512 of them take about 28 seconds across 128 cores.

The HBPLUS-dependent protonation labeller is removed from the featurisation pipeline. Without it the
native sequence falls back to canonical `ASP`/`GLU`/`HIS` tokens, which is in distribution: the
checkpoint trained at `protonation_label_rate=0.5`, so half of training saw exactly those. The Potts
energy table is a function of backbone geometry alone, so the labels never reached it.
