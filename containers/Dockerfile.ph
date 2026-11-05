# BC2 plus the Proton-PottsMPNN pH-switch redesign stage, as one image that still runs vanilla.
#
#   docker build -t bindcraft2:ph -f containers/Dockerfile.ph .
#   docker run --gpus all -v "$PWD:/work" bindcraft2:ph design settings.json
#
# With `ph_redesign` absent or false from the settings file this image behaves exactly as the base:
# bindcraft/ph_redesign.py is never imported and nothing below is on the campaign's import path.
#
# The pH engine lives in its own interpreter under /opt/protonpotts/venv because atomworks pins
# biotite==1.4.0 while BindCraft2 needs 1.7.x. That is the whole reason for the venv; the two cannot
# share one site-packages. The stage is a short CPU batch next to a ~18 s featurisation, so the
# subprocess hop costs nothing measurable.
#
# torch is installed from the CPU index on purpose. The pH engine cannot use a GPU anyway -- its fork
# pool is gated on device == "cpu", and 512 designs take ~28 s on CPU -- and a CUDA torch would drop a
# second set of nvidia/*/lib directories into the base image's /etc/ld.so.conf.d/bindcraft-cuda.conf,
# where ldconfig would pick one cuDNN globally for both frameworks and could silently break jax.
ARG BASE=bindcraft2:latest
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Vendored before the install so the expensive layer survives edits to the bindcraft package below.
# Only what the worker uses; the submodule also carries upstream's training, scoring and figures.
COPY vendor/ProtonPottsMPNN/foundry /opt/protonpotts/foundry
COPY vendor/ProtonPottsMPNN/checkpoints /opt/protonpotts/checkpoints
COPY vendor/ProtonPottsMPNN/inference/examples /opt/protonpotts/examples

# hatch-vcs derives the version from git, and there is no .git here.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 HATCH_VCS_PRETEND_VERSION=0.0.0
RUN python3 -m venv /opt/protonpotts/venv \
 && /opt/protonpotts/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/protonpotts/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && /opt/protonpotts/venv/bin/pip install --no-cache-dir /opt/protonpotts/foundry "propka>=3.5" ipdb "pandas>=2.1,<3" \
 && ! /opt/protonpotts/venv/bin/python -c "import nvidia" 2>/dev/null \
 && /opt/protonpotts/venv/bin/python -c "import torch, biotite; assert '+cpu' in torch.__version__, torch.__version__; print('ph venv: torch', torch.__version__, 'biotite', biotite.__version__)"

COPY vendor/ph_worker.py /opt/protonpotts/ph_worker.py

# Only the package and the settings tree, so /opt/protonpotts is not duplicated under /opt/bindcraft.
# Changing anything else at the repository root (scaffolds/, pyproject.toml) needs a base rebuild.
COPY bindcraft /opt/bindcraft/bindcraft
COPY settings /opt/bindcraft/settings

# Fails the build rather than a campaign: the base image must still import clean with no torch in its
# path, every attribute the campaign reads off the sequence model must survive the wrapper, and the pH
# worker must round-trip the shipped example end to end with no HBPLUS present.
COPY containers/ph_proxy_check.py /opt/ph_proxy_check.py
WORKDIR /
RUN python3 -c "import bindcraft.campaign, bindcraft.proteinmpnn; import importlib.util as u; assert u.find_spec('torch') is None, 'torch leaked into the bindcraft interpreter'; print('bindcraft interpreter clean')" \
 && python3 -c "import bindcraft.ph_redesign as p; print('ph wrapper imports:', p.PHRedesignSequenceModel.__name__)" \
 && python3 /opt/ph_proxy_check.py \
 && printf '%s' '{"structure":"/opt/protonpotts/examples/pdl1_seed_binder.pdb","binder_chain":"A","seeds":[],"criteria":{"lambdas":[1.0]},"n_jobs":1,"seed":0}' > /tmp/ph_job.json \
 && { /opt/protonpotts/venv/bin/python /opt/protonpotts/ph_worker.py /tmp/ph_job.json /tmp/ph_result.json \
      || { python3 -c "import json; r=json.load(open('/tmp/ph_result.json')); print(r.get('error', 'no error recorded')); print(r.get('traceback', ''))"; exit 1; }; } \
 && python3 -c "import json; r=json.load(open('/tmp/ph_result.json')); print('ph smoke test:', len(r['designs']), 'design(s), selective', r['designs'][0]['selective_energy'])" \
 && rm -f /tmp/ph_job.json /tmp/ph_result.json

WORKDIR /work
ENTRYPOINT ["bindcraft"]
CMD ["--help"]
