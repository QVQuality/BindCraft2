"""Build gate: every attribute the campaign reads off the sequence model must survive the pH wrapper.

A missing one is not caught until metadata stamping, which runs after a design is accepted, so it
discards work that is already built and refolded. The required set is read out of the source rather
than hardcoded, so adding a new `mpnn_model.<attr>` read anywhere in the package fails the build
instead of a campaign.
"""
import os
import re
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

METHODS = {'predict', 'predict_candidates', 'predict_tied_candidates'}


def package_directory() -> str:
    """Where the installed bindcraft package's sources are.

    Resolved from a known submodule's spec: the editable install can present the top-level package
    as a namespace package, where __file__ is None and __path__ is not always populated the same way
    at build time as at run time.
    """
    import importlib.util
    for module in ('bindcraft.campaign', 'bindcraft.campaign_output', 'bindcraft.proteinmpnn'):
        spec = importlib.util.find_spec(module)
        if spec is not None and spec.origin:
            directory = os.path.dirname(os.path.abspath(spec.origin))
            if os.path.isdir(directory):
                return directory
    import bindcraft
    for candidate in list(getattr(bindcraft, '__path__', []) or []):
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError('cannot locate the installed bindcraft package sources')


def attributes_the_campaign_reads() -> set[str]:
    package = package_directory()
    found: set[str] = set()
    for entry in sorted(os.listdir(package)):
        if not entry.endswith('.py'):
            continue
        with open(os.path.join(package, entry)) as handle:
            found.update(re.findall(r'\bmpnn_model\.([A-Za-z_][A-Za-z0-9_]*)', handle.read()))
    return found - METHODS


def main() -> int:
    from bindcraft.model_weights import proteinmpnn_weights
    from bindcraft.proteinmpnn import ProteinMPNNSequenceModel
    from bindcraft.ph_redesign import PHRedesignSequenceModel

    required = attributes_the_campaign_reads()
    # A gate that finds nothing passes everything, so refuse to be vacuous. These three are read by
    # campaign_output.reprediction_facts and are the reason this check exists.
    if not {'model_name', 'variant', 'temperature'} <= required:
        print(f'FAIL: the check found only {sorted(required)} in {package_directory()}; it is not '
              'seeing the package sources and would pass vacuously', file=sys.stderr)
        return 1
    inner = ProteinMPNNSequenceModel(data_dir=proteinmpnn_weights(), variant='negative')
    wrapper = PHRedesignSequenceModel(inner)

    missing = []
    for name in sorted(required):
        try:
            getattr(wrapper, name)
        except AttributeError:
            missing.append(name)
    print(f'ph proxy surface: {len(required)} attribute(s) read by the campaign '
          f'({", ".join(sorted(required))})')
    if missing:
        print(f'FAIL: the pH wrapper cannot serve {", ".join(missing)}', file=sys.stderr)
        return 1

    # The exact call that discarded accepted designs.
    from bindcraft.campaign_output import reprediction_facts

    class Validation:
        num_recycle = 3

    facts = reprediction_facts(wrapper, Validation())
    assert facts['mpnn_model'] and facts['mpnn_variant'] and facts['mpnn_temperature'], facts
    print(f'reprediction_facts over the wrapper: {facts}')

    # Delegation must not swallow the protocol lookups copy and pickle make.
    for dunder in ('__deepcopy__', '__reduce_ex__', '__getstate__'):
        try:
            getattr(wrapper, dunder)
        except AttributeError:
            pass
    assert not hasattr(wrapper, 'definitely_not_an_attribute'), 'delegation is too permissive'
    print('ph proxy check: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
