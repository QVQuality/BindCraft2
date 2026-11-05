"""pH-switch sequence redesign, layered on the ProteinMPNN stage.

ProteinMPNN draws the sequence candidates as usual; Proton-PottsMPNN then pins a protonated centre
and redesigns its neighbourhood so the binder's energy differs between the centre's protonated and
deprotonated states. The gap is the switch: protonation follows pH, so a design whose interface is
favourable only when the centre carries a proton binds at low pH and releases at neutral.

The engine runs in a separate interpreter (/opt/protonpotts/venv) because atomworks pins
biotite==1.4.0 while this package needs 1.7.x. With `ph_redesign` unset nothing here is imported and
the campaign behaves exactly as stock BindCraft2.
"""
import json
import os
import subprocess
import tempfile

import jax.numpy as jnp
import numpy as np
from jax import Array

from bindcraft.prediction import ProteinPredictor
from bindcraft.protein import (
    AMINO_ACIDS,
    Protein,
    ProteinStates,
    ResidueFlags,
    StructurePrediction,
    THREE_LETTER_CODE,
    StructurePredictions,
    has_residue_flag,
    is_binder_chain,
    output_chain_letters,
    write_structure,
)

PH_WORKER = os.environ.get('BINDCRAFT_PH_WORKER', '/opt/protonpotts/ph_worker.py')
PH_PYTHON = os.environ.get('BINDCRAFT_PH_PYTHON', '/opt/protonpotts/venv/bin/python')
_ONE_LETTER_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def one_hot_sequence(sequence: str) -> Array:
    indices = np.asarray([_ONE_LETTER_INDEX.get(residue, _ONE_LETTER_INDEX['A']) for residue in sequence])
    return jnp.asarray(np.eye(len(AMINO_ACIDS), dtype=np.float16)[indices])


def interface_contact_mask(protein_complex: dict[str, Protein], binder: str, cutoff: float=4.0) -> Array:
    """Binder positions touching any target chain, by the same heavy-atom test the stage holds on."""
    from bindcraft.filters import binder_target_contact_masks
    contacts = jnp.zeros(len(protein_complex[binder]), dtype=bool)
    for name, protein in protein_complex.items():
        if not is_binder_chain(name):
            contacts = contacts | binder_target_contact_masks(protein_complex[binder], protein, cutoff)[0]
    return contacts


def held_residue_ids(protein_complex: dict[str, Protein], release_interface: bool=False) -> list[str]:
    """Every position the pH pass must not touch, as '<chain letter><residue number>'.

    That is the whole target plus whatever the stage held on the binder (a kept interface, a
    fold-conditioning framework, a multi-domain linker, an induced-fit mobile set).

    Everything travels as fixed_residues, including the target, for two reasons. The designed_* fields
    are applied last by MPNNInferenceInput._annotate_design_scope and would re-open every hold, and
    _validate_design_scope refuses to mix residue-based with chain-based constraints, so a
    fixed_chains shortcut for the target cannot coexist with per-residue holds.
    """
    letters = output_chain_letters(protein_complex)
    held: list[str] = []
    for chain_name, protein in protein_complex.items():
        residue_index = np.asarray(protein.residue_index)
        if is_binder_chain(chain_name):
            design_mask = np.asarray(has_residue_flag(protein.flags, ResidueFlags.DESIGN))
            # redesign_interface defaults false, so the stage hands the whole paratope over held, which
            # is exactly what a pH centre needs to perturb. Releasing it here lets ProteinMPNN keep
            # preserving the interface while the pH pass alone is allowed to touch it.
            if release_interface:
                design_mask = design_mask | np.asarray(
                    interface_contact_mask(protein_complex, chain_name))
            positions = np.flatnonzero(~design_mask)
        else:
            positions = np.arange(len(protein))
        held += [f'{letters[chain_name]}{int(residue_index[position])}' for position in positions]
    return held


def binder_chain_letter(protein_complex: dict[str, Protein]) -> str:
    letters = output_chain_letters(protein_complex)
    return next(letters[name] for name in sorted(protein_complex) if is_binder_chain(name))


def primary_binder_chain(protein_complex: dict[str, Protein]) -> str:
    return next(name for name in sorted(protein_complex) if is_binder_chain(name))


def run_ph_worker(job: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix='bindcraft-ph-') as workspace:
        job_path = os.path.join(workspace, 'job.json')
        result_path = os.path.join(workspace, 'result.json')
        structure_path = os.path.join(workspace, 'complex.pdb')
        write_structure(job.pop('_protein_complex'), structure_path)
        job['structure'] = structure_path
        with open(job_path, 'w') as handle:
            json.dump(job, handle)
        completed = subprocess.run([PH_PYTHON, PH_WORKER, job_path, result_path],
                                   capture_output=True, text=True, timeout=timeout)
        if not os.path.exists(result_path):
            return {'designs': [], 'error': f'worker produced no result (exit {completed.returncode})',
                    'traceback': completed.stderr[-4000:]}
        with open(result_path) as handle:
            return json.load(handle)


def inner_omissions(mpnn_model) -> str:
    """Amino acids ProteinMPNN was told to exclude, recovered from the bias array it enforces.

    ProteinMPNNSequenceModel folds omitted_amino_acids into redesigned_amino_acid_bias and keeps no
    copy of the string, so reading the array is what cannot drift from what is actually applied.
    """
    from bindcraft.sequence_optimization import omitted_amino_acid_mask
    bias = getattr(mpnn_model, 'redesigned_amino_acid_bias', None)
    if bias is None:
        return getattr(mpnn_model, 'omitted_amino_acids', '') or ''
    excluded = np.asarray(omitted_amino_acid_mask(bias))
    return ''.join(AMINO_ACIDS[index] for index in np.flatnonzero(excluded))


class PHRedesignSequenceModel(ProteinPredictor):
    """ProteinMPNN, then a pH-switch redesign pass over its candidates.

    Satisfies the same protocol the redesign stage uses, so MPNN_stage.py is untouched: it calls only
    predict_candidates and predict_tied_candidates.
    """

    def __init__(self, mpnn_model, criteria: dict | None=None, n_jobs: int=8,
                 timeout: float=1800.0, log_path: str | None=None, strict: bool=False,
                 redesign_interface: bool=False, rank: str='combined', give_up_after: int=5,
                 omitted_amino_acids: str=''):
        self.mpnn_model = mpnn_model
        self.criteria = dict(criteria or {})
        self.n_jobs = int(n_jobs)
        self.timeout = float(timeout)
        self.log_path = log_path or os.environ.get('BINDCRAFT_PH_LOG')
        self.strict = bool(strict)
        self.redesign_interface = bool(redesign_interface)
        self.rank = str(rank)
        self.give_up_after = int(give_up_after)
        self.failures = 0
        self.consecutive_empty = 0
        # A propensity of zero becomes an omission upstream (settings.resolve_omitted_amino_acids), and
        # the pH pass would otherwise reintroduce the residue while rewriting the neighbourhood.
        # Read off the bias array ProteinMPNN actually enforces rather than a separate setting, because
        # ProteinMPNNSequenceModel consumes omitted_amino_acids into that array and keeps no copy.
        self.extra_forbidden_tokens = [THREE_LETTER_CODE[residue]
                                       for residue in (omitted_amino_acids or inner_omissions(mpnn_model))
                                       if residue in THREE_LETTER_CODE]
        if self.extra_forbidden_tokens:
            print(f'ph_redesign: forbidding {", ".join(self.extra_forbidden_tokens)} to match the '
                  "campaign's omitted amino acids", flush=True)

    def __getattr__(self, name: str):
        """Anything this wrapper does not define is answered by the ProteinMPNN it wraps.

        The campaign reads model_name, variant and temperature off the sequence model when it stamps
        metadata, and that happens *after* a design is accepted: a missing attribute throws away a
        design that is already built and refolded. Delegating keeps the wrapper transparent to
        whatever the campaign reads next instead of fixing one attribute at a time.

        Only reached when normal lookup fails. Underscored names are refused so the copy and pickle
        protocols are not captured, and the wrapped model is fetched without going through this method
        so a lookup before __init__ finishes cannot recurse.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            wrapped = object.__getattribute__(self, 'mpnn_model')
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(wrapped, name)

    def predict(self, protein_states: ProteinStates, model: str | None=None) -> StructurePredictions:
        return self.predict_candidates(protein_states, model)[0]

    def predict_candidates(self, protein_states: ProteinStates, model: str | None=None,
                           candidate_count: int=1) -> list[StructurePredictions]:
        seeded = self.mpnn_model.predict_candidates(protein_states, model, candidate_count)
        return self._redesign(protein_states, seeded)

    def predict_tied_candidates(self, protein_states: ProteinStates,
                                candidate_count: int=1) -> list[StructurePredictions]:
        seeded = self.mpnn_model.predict_tied_candidates(protein_states, candidate_count)
        return self._redesign(protein_states, seeded)

    def _redesign(self, protein_states: ProteinStates,
                  seeded: list[StructurePredictions]) -> list[StructurePredictions]:
        """Redesign on one state's backbone and share the sequence across the rest.

        Block descent has no tied-decoding equivalent, and the stage shares the binder sequence across
        tied chains and states anyway (share_binder_chain_sequences), so one pass is the honest unit.
        """
        if not seeded:
            return seeded
        reference_state = sorted(protein_states)[0]
        reference_complex = seeded[0][reference_state].protein_complex
        binder = primary_binder_chain(reference_complex)
        seeds = [self._binder_sequence(candidate[reference_state].protein_complex, binder)
                 for candidate in seeded]

        held = held_residue_ids(reference_complex, release_interface=self.redesign_interface)
        designable = sum(len(protein) for name, protein in reference_complex.items()
                         if is_binder_chain(name)) - sum(
            1 for identifier in held if identifier.startswith(binder_chain_letter(reference_complex)))
        if designable <= 0:
            # Featurising a complex with nothing designable has been seen to hang the engine rather
            # than return, so it is refused here instead of waiting out ph_timeout.
            self.failures += 1
            message = ('ph_redesign: no binder position is designable, so there is nothing to '
                       'redesign around a centre; keeping ProteinMPNN sequences')
            if self.strict:
                raise RuntimeError(message)
            print(message, flush=True)
            return seeded
        result = run_ph_worker({
            '_protein_complex': reference_complex,
            'binder_chain': binder_chain_letter(reference_complex),
            'fixed_residues': held,
            'extra_forbidden_tokens': self.extra_forbidden_tokens,
            'seeds': seeds,
            'criteria': self.criteria,
            'n_jobs': self.n_jobs,
            'seed': 0,
        }, self.timeout)

        # An empty design list carries no error: placement can simply find nowhere to put a centre.
        # Treated as success it would run the campaign as stock BindCraft2 and leave an empty log as
        # the only evidence, so it is a failure here and `ph_strict` catches it.
        if result.get('error') or not result.get('designs'):
            self.failures += 1
            self.consecutive_empty += 1
            reason = result.get('error') or ('worker pinned no centre; with '
                                             'placement_region=["interface"] check the paratope is '
                                             'designable (ph_redesign_interface) and in contact')
            message = f'ph_redesign: {reason}; keeping ProteinMPNN sequences'
            if self.strict:
                raise RuntimeError(f"{message}\n{result.get('traceback', '')}")
            print(message, flush=True)
            if self.give_up_after and self.consecutive_empty == self.give_up_after:
                print(f'ph_redesign: {self.consecutive_empty} batches in a row pinned nothing; the '
                      'stage is a no-op as configured and the campaign is running as stock '
                      'BindCraft2', flush=True)
            return seeded
        self.consecutive_empty = 0

        best = self._best_per_seed(result['designs'], len(seeded))
        kept = 0
        redesigned: list[StructurePredictions] = []
        for index, candidate in enumerate(seeded):
            design = best.get(index)
            if design is None:
                redesigned.append(candidate)
                continue
            kept += 1
            self._record(design, index, seeds[index])
            redesigned.append(self._apply(candidate, design['sequence']))
        print(f'ph_redesign: {kept}/{len(seeded)} candidate(s) carry a pinned protonated centre'
              f" ({result.get('designs') and len(result['designs'])} design(s) returned)", flush=True)
        return redesigned

    def _best_per_seed(self, designs: list[dict], candidate_count: int) -> dict[int, dict]:
        """Pick one design per seed. Lower is better for both energies.

        Ranking on selective energy alone makes a lambda sweep pointless: the lambda=1 member wins by
        construction, because it is the only one that put all the weight there, so the sweep collapses
        onto its least stable arm. 'combined' scores each design on the same objective its own criteria
        optimised, (1-lambda)*z(stability) + lambda*z(selective), with the z-scores taken over this
        batch so the two energies are comparable.
        """
        score = self._scorer(designs)
        best: dict[int, dict] = {}
        for design in designs:
            seed_index = design.get('seed_idx')
            seed_index = 0 if seed_index is None else int(seed_index)
            if seed_index >= candidate_count:
                continue
            incumbent = best.get(seed_index)
            if incumbent is None or score(design) < score(incumbent):
                best[seed_index] = design
        return best

    def _scorer(self, designs: list[dict]):
        if self.rank == 'selective':
            return lambda design: _finite(design.get('selective_energy'))
        if self.rank == 'stability':
            return lambda design: _finite(design.get('potts_energy'))
        selective = _standardizer([_finite(d.get('selective_energy')) for d in designs])
        stability = _standardizer([_finite(d.get('potts_energy')) for d in designs])

        def combined(design: dict) -> float:
            weight = design.get('combined_lambda')
            weight = 1.0 if weight is None else float(weight)
            return ((1.0 - weight) * stability(_finite(design.get('potts_energy')))
                    + weight * selective(_finite(design.get('selective_energy'))))
        return combined

    def _binder_sequence(self, protein_complex: dict[str, Protein], binder: str) -> str:
        indices = np.asarray(protein_complex[binder].sequence.argmax(-1))
        return ''.join(AMINO_ACIDS[int(index)] for index in indices)

    def _apply(self, candidate: StructurePredictions, sequence: str) -> StructurePredictions:
        updated: StructurePredictions = {}
        for state_name, prediction in candidate.items():
            protein_complex = dict(prediction.protein_complex)
            for chain_name, protein in protein_complex.items():
                if not is_binder_chain(chain_name):
                    continue
                if len(sequence) != len(protein):
                    continue
                protein_complex[chain_name] = protein.replace(sequence=one_hot_sequence(sequence))
            updated[state_name] = StructurePrediction(protein_complex=protein_complex,
                                                      metrics=prediction.metrics)
        return updated

    def _record(self, design: dict, candidate_index: int, seed_sequence: str) -> None:
        if not self.log_path:
            return
        mutations = sum(1 for seed, designed in zip(seed_sequence, design.get('sequence') or '')
                        if seed != designed)
        row = {'candidate': candidate_index + 1, 'seed_sequence': seed_sequence,
               'mutations_vs_seed': mutations,
               **{key: design.get(key) for key in (
            'design_id', 'sequence', 'potts_energy', 'selective_energy', 'global_protonation_dH',
            'combined_lambda', 'center_res_ids', 'center_protonation_types', 'n_neighbours')}}
        os.makedirs(os.path.dirname(self.log_path) or '.', exist_ok=True)
        with open(self.log_path, 'a') as handle:
            handle.write(json.dumps(row) + '\n')


def _finite(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float('inf')
    return value if value == value else float('inf')


def _standardizer(values: list[float]):
    """z-score against the batch; a degenerate spread collapses to 0 so the other term decides."""
    usable = [value for value in values if value != float('inf')]
    if not usable:
        return lambda value: 0.0
    mean = sum(usable) / len(usable)
    spread = (sum((value - mean) ** 2 for value in usable) / len(usable)) ** 0.5
    if spread <= 0.0:
        return lambda value: 0.0
    return lambda value: 0.0 if value == float('inf') else (value - mean) / spread
