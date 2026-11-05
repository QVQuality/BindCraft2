"""pH-switch redesign worker, run inside /opt/protonpotts/venv.

Reads a job JSON, runs Proton-PottsMPNN block descent over one backbone, writes a result JSON.
Lives in its own interpreter because atomworks pins biotite==1.4.0 while BindCraft2 needs 1.7.x.

    python ph_worker.py job.json result.json
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("DEBUG", None)

CHECKPOINT = os.environ.get(
    "PROTONPOTTS_CHECKPOINT",
    "/opt/protonpotts/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt",
)

# The infill may otherwise introduce ASP-P/GLU-P centres that were never asked for. Their side-chain
# pKa is ~3.9/4.3, so a neutral carboxyl is unpopulated at pH 6.5 and the selectivity it buys cannot
# operate in a 6.5/7.4 window. Measured on the shipped PD-L1 example: 5 spurious ASP-P centres and
# ~36% of the apparent selective gap.
FORBID_EXTRA_PROTONATED = ["ASP-P", "GLU-P"]
BASE_FORBIDDEN = ["HIS-A", "ASP-A", "GLU-A", "HIS-D"]

# The objective is O = (1-lambda)*z(H_stab) + lambda*z(selective), so lambda=1 is pure selectivity
# with zero weight on the fold. Measured on an 86-mer EGFR design, lambda=1 rewrote 12 of 86
# positions and replaced a helix with an acidic block, where 0.3 moved 5 and left it recognisable.
# 0.3 is also what the upstream example uses.
DEFAULT_LAMBDA = 0.3

# Only the backstop for the heavy-atom interface test below, never the primary definition. The
# engine's own default of 6.0 reports an empty interface on a side-chain-mediated one.
PROXY_FALLBACK_DIST = 8.0


def _patch_region_masks(contact_distance):
    """Define `interface` by heavy-atom contact instead of the engine's CA-CA proxy.

    _binder_region_masks calls a binder residue interfacial only if its CA is within 6.0 A of a target
    CA, which a side-chain-mediated interface fails: on an 86-mer from an EGFR campaign the closest
    CA-CA approach is 6.03 A, so the mask is empty while 15 residues are in heavy-atom contact at
    2.05 A. placement_region=["interface"] then has nowhere to put a centre and returns no designs.

    Raising the CA-CA cutoff would paper over it at a value that is specific to one fold and one
    target. Heavy-atom contact is what "interface" means, and it is also what BindCraft's own
    binder_target_contact_masks uses, so the two layers now agree by construction.
    """
    import numpy as np
    import mpnn.inference_engines.potts_mpnn_ph as engine_module

    original = engine_module._binder_region_masks

    def heavy_atom_interface(proc_atom_array, token_aa, binder_chain, chainA_t, device, length,
                             *args, **kwargs):
        # Widen the engine's own CA-CA cutoff as well, so the fallback below is usable rather than
        # empty if the heavy-atom pass ever fails. 6.0 is unusable on a side-chain interface: the
        # 86-mer's closest CA-CA approach is 6.03 A. This value is only ever the backstop.
        kwargs.setdefault("interface_dist", PROXY_FALLBACK_DIST)
        masks = original(proc_atom_array, token_aa, binder_chain, chainA_t, device, length,
                         *args, **kwargs)
        try:
            import torch
            atoms = proc_atom_array
            binder_side = np.asarray(atoms.chain_id == binder_chain)
            target_side = ~binder_side
            if not binder_side.any() or not target_side.any():
                return masks
            on_binder = np.asarray(chainA_t.detach().cpu().numpy())
            position_of = {(str(token_aa.chain_id[i]), int(token_aa.res_id[i])): i
                           for i in range(length) if on_binder[i]}
            target_coords = atoms.coord[target_side]
            interface = torch.zeros(length, dtype=torch.bool, device=device)
            binder_indices = np.flatnonzero(binder_side)
            separations = np.linalg.norm(
                atoms.coord[binder_indices][:, None, :] - target_coords[None, :, :], axis=-1)
            contacting = binder_indices[separations.min(axis=1) < float(contact_distance)]
            for atom_index in contacting:
                position = position_of.get((str(atoms.chain_id[atom_index]),
                                            int(atoms.res_id[atom_index])))
                if position is not None:
                    interface[position] = True
            proxy_count = int(masks["interface"].sum())
            masks["interface"] = interface
            print(f"ph_worker: interface = {int(interface.sum())} residue(s) by heavy-atom contact "
                  f"< {contact_distance} A (the CA-CA proxy found {proxy_count})", flush=True)
        except Exception as failure:
            print(f"ph_worker: heavy-atom interface unavailable ({failure}); keeping the CA-CA proxy",
                  flush=True)
        return masks

    engine_module._binder_region_masks = heavy_atom_interface


def _patch_center_resolution():
    """Let an explicit centre name a residue of the fixed target, not just the binder.

    enumerate_placement_plans resolves every explicit_centers entry through
    _PHContext.binder_pos_of_res_id, whose res_id_to_pos map is built from binder positions only, so
    pinning a target residue raises. Nothing downstream needs the pin to be designable:
    _finalize_plan only requires a non-empty neighbourhood, and neighbour_mask walks the coupling
    graph keeping free BINDER positions, which is exactly the wanted shape -- pin the target's
    titratable residue, redesign the binder around it.

    The binder lookup is tried first, so a res_id present on both chains still resolves to the binder
    and existing behaviour is unchanged.
    """
    import mpnn.inference_engines.potts_mpnn_ph as engine_module

    context_class = engine_module._PHContext
    original = context_class.binder_pos_of_res_id

    def position_of_res_id(self, res_id: int) -> int:
        try:
            return original(self, res_id)
        except ValueError:
            matches = [position for position in range(self.L)
                       if int(self.token_aa.res_id[position]) == int(res_id)]
            if not matches:
                raise
            position = matches[0]
            print(f"ph_worker: centre res_id {res_id} resolved to chain "
                  f"{self.token_aa.chain_id[position]} position {position} (not the binder); the "
                  "binder neighbourhood will be redesigned around it", flush=True)
            return position

    context_class.binder_pos_of_res_id = position_of_res_id


def _patch_pipeline(fixed_residues):
    """Drop the HBPLUS/labeller block and thread per-residue holds into featurisation.

    Without the protonation block no `protonation_label` annotation is produced, and S falls back to
    canonical ASP/GLU/HIS. That path is in-distribution: the checkpoint trained at
    protonation_label_rate=0.5, so half of training saw canonical tokens.

    The holds express themselves as fixed_* and REPLACE the engine's hardcoded
    designed_chains=[binder]. _annotate_design_scope applies the designed_* fields last, so leaving
    designed_chains in place would re-open every held position; with only fixed_residues set the mask
    starts all-designable, which is what makes "binder minus holds" come out right. The target travels
    in the same list because _validate_design_scope refuses to mix residue- and chain-based scopes.
    """
    import mpnn.pipelines.potts_mpnn as pipelines

    pipelines.get_protonation_state_transforms = lambda **kw: [pipelines.RemoveHydrogens()]

    import mpnn.inference_engines.potts_mpnn_ph as engine_module

    original = engine_module.prepare_potts_input

    def with_holds(structure, **kwargs):
        if fixed_residues:
            kwargs.pop("designed_chains", None)
            kwargs["fixed_residues"] = list(fixed_residues)
        return original(structure, **kwargs)

    engine_module.prepare_potts_input = with_holds


def run(job):
    spec = dict(job.get("criteria") or {})
    _patch_region_masks(float(spec.pop("interface_contact_distance", 4.5)))
    if spec.pop("allow_target_centers", True):
        _patch_center_resolution()
    _patch_pipeline(job.get("fixed_residues") or [])

    from biotite.structure.io.pdb import PDBFile
    from mpnn.inference_engines.potts_mpnn_ph import PHDesignCriteria, PottsMPNNPHEngine

    atom_array = PDBFile.read(job["structure"]).get_structure(model=1)
    engine = PottsMPNNPHEngine(
        checkpoint_path=job.get("checkpoint", CHECKPOINT),
        extended_vocab="v6",
        out_directory=None,
        write_fasta=False,
        write_structures=False,
    )

    forbidden = spec.pop("forbidden_tokens", None)
    forbid_extra = bool(spec.pop("forbid_extra_protonated", True))
    if forbidden is None:
        forbidden = BASE_FORBIDDEN + (FORBID_EXTRA_PROTONATED if forbid_extra else [])
    # Residues the campaign excluded outright (aa_bias weight <= 0 becomes omitted_amino_acids), as
    # canonical three-letter tokens. Without these the pH pass would reintroduce a residue the
    # campaign forbids; a zero propensity means excluded, not unbiased.
    forbidden = list(dict.fromkeys(list(forbidden) + list(job.get("extra_forbidden_tokens") or [])))

    # combined_lambda is popped unconditionally: left in spec it collides with the explicit keyword
    # below and raises TypeError for anyone who passes both it and lambdas.
    single = spec.pop("combined_lambda", None)
    lambdas = spec.pop("lambdas", None) or ([single] if single is not None else [DEFAULT_LAMBDA])
    seeds = job.get("seeds") or []

    defaults = dict(
        method="block_descent",
        backend="potts",
        center_types=["HIS-P"],
        dep_map={"HIS-P": ["HIS-S"]},
        placement_by="scan_potts",
        placement_region=["interface"],
        block_size=3,
        temperature=0.05,
        neighbour_k=16,
        max_mutations=20,
        record_trajectory=False,
    )
    defaults.update(spec)
    defaults["seed_source"] = "inverse" if seeds else "native"
    defaults["forbidden_tokens"] = list(forbidden)

    criteria = [PHDesignCriteria(combined_lambda=float(lam), **defaults) for lam in lambdas]
    designs = engine.run_ph_redesign(
        atom_array=atom_array,
        binder_chain=job["binder_chain"],
        criteria_list=criteria,
        seed=int(job.get("seed", 0)),
        initial_sequences=seeds or None,
        n_jobs=int(job.get("n_jobs", 1)),
    )

    out = []
    for design in designs:
        meta = design.to_metadata()
        out.append(
            {
                "sequence": design.canonical_sequence,
                "extended_tokens": list(design.extended_tokens),
                "potts_energy": _finite(design.final_potts_energy),
                "selective_energy": _finite(design.selective_energy),
                "global_protonation_dH": _finite(design.global_protonation_dH),
                "design_id": meta.get("design_id"),
                "seed_idx": design.seed_idx,
                "combined_lambda": design.combined_lambda,
                "center_res_ids": design.center_res_ids,
                "center_protonation_types": design.center_protonation_types,
                "n_neighbours": design.n_neighbours,
            }
        )
    return {"designs": out, "forbidden_tokens": list(forbidden)}


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def main():
    job = json.loads(open(sys.argv[1]).read())
    try:
        result = run(job)
    except Exception as failure:
        import traceback

        result = {"designs": [], "error": f"{type(failure).__name__}: {failure}",
                  "traceback": traceback.format_exc()}
    with open(sys.argv[2], "w") as handle:
        json.dump(result, handle)
    return 0 if result.get("designs") or not result.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
