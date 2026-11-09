# SynThera

SynThera is a machine learning pipeline for designing therapeutics against high consequence pathogens. In one Design, Build, Test, Learn (DBTL) loop it predicts potency, generates retrobiosynthetic routes, ranks candidates by ADMET (absorption, distribution, metabolism, excretion, toxicity), filters routes for thermodynamic feasibility, and designs DNA for the production host. The repository holds the notebooks, scripts, environment specifications and Docker build needed to reproduce it.

> **Read-only showcase.** This copy holds the code, notebooks, environment specifications and Docker build of SynThera. The Ebola case study notebooks (Step1 to Step5) keep their saved outputs. The data files, saved models and the container image are not included, so the pipeline cannot be rerun from this repository. Commands below that refer to the image use the placeholders `<REGISTRY>/<IMAGE>:<TAG>`.

## Pipeline

Each pathogen case study runs five notebooks in order:

1. `Step1_DataPreparation`: assemble and clean bioactivity training data.
2. `Step2_BuildModel`: build molecular embeddings and train potency predictors, including ART (Automated Recommendation Tool) models.
3. `Step3_GenerateCompounds`: expand starting materials into candidates with DORAnet and RetroTide, then filter for drug likeness.
4. `Step4_PredictProperties`: score candidates for potency and ADMET.
5. `Step5_ExperimentalDesigns`: turn the top candidates into DNA designs for the production host.

## Repository layout

The case study folders and data folders of the original project are not part of this copy.

| Path | Contents |
|---|---|
| `Step0_GetBioactivityData.ipynb`, `Step1_DataPreparation.ipynb` | Shared data assembly, run once. |
| `Step2_BuildModel.ipynb` to `Step5_DesignExperiment.ipynb` (root) | The Ebola case study notebooks with their saved outputs. |
| `executables/` | Scripts for the long steps (`runDORAnet.py`, `runDORAXGB.py`, `runADMET.py`, `runART.py`), each with a `*_config.yaml` template. |
| `external/ergochemics/` | Bundled reaction fingerprinting library used by Step5 (example data files removed). |
| `tests/` | Build checks. See [Verifying your build](#verifying-your-build). |
| `build/` | Dockerfile, environment specs and Docker documentation. |

## Quick start: Docker

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS, Windows) or Docker Engine (Linux) and start it. Then pull the image and run it from the repository root, so your notebooks and data are mounted in the container:

```bash
docker pull <REGISTRY>/<IMAGE>:<TAG>
docker run -p 8888:8888 -v "$(pwd)":/opt/synthera <REGISTRY>/<IMAGE>:<TAG>
```

To build the image yourself instead (it is over 10 GB because it pins CUDA builds of torch and tensorflow):

```bash
cd build
docker compose up --build
```

Open the `http://127.0.0.1:8888/lab?token=...` URL printed in the terminal. Stop the container with `docker compose down`. Your files stay in this folder.

The image is `linux/amd64`, so on Apple Silicon it runs under emulation and is slower. GPU setup and other details are in [`build/DOCKER.md`](build/DOCKER.md).

### Kernels

The image has two conda environments. Pick the kernel that matches the notebook (Kernel menu, Change Kernel).

| Notebook | Kernel |
|---|---|
| Step0, Step1, Step2, Step4 | `SynThera` (`build/environment.yml`) |
| Step3, Step5 | `SynThera_ExptDesigns` (`build/exptDesigns_environment.yml`) |

### Installing ART (required for Step2 and Step4)

Step2 and Step4 need ART. It is free for academic, nonprofit and U.S. government use but is licensed separately, so the image does not include it. Install it before running either step, because without it `import art` fails. See [`build/ART.md`](build/ART.md) for the license, then build with:

```bash
docker build --build-arg ART_SOURCE="git+https://github.com/JBEI/AutomatedRecommendationTool.git" \
  -f build/Dockerfile -t synthera:with-art .
```

To install into a running container, use `build/installART.sh`.

## Verifying your build

The image holds its own copy of `executables/`, `tests/` and `external/` in `/opt/synthera-app`, so you can test it without a source checkout:

```bash
docker run --rm <image> bash /opt/synthera-app/tests/run_all_tests.sh          # a few seconds
docker run --rm <image> bash /opt/synthera-app/tests/run_all_tests.sh --full   # real runs, several minutes, needs network
```

Use `<REGISTRY>/<IMAGE>:<TAG>` or your own tag for `<image>`. Each check prints `[PASS]`, `[FAIL]` or `[SKIP]`. A `[SKIP]` for ART means ART is not installed in that image. Details are in [`tests/README.md`](tests/README.md).

## Running the long compute steps

DORAnet searches, ADMET scoring, DORA-XGB ranking and ART cross validation take too long for a notebook cell, so they run as scripts. The notebook writes a YAML config next to its results. Run the script on that config from a JupyterLab terminal (File, New, Terminal) or with `docker exec -it synthera bash`:

```bash
conda run -n SynThera             python executables/runADMET.py   <config>.yaml
conda run -n SynThera             python executables/runART.py     <config>.yaml
conda run -n SynThera_ExptDesigns python executables/runDORAnet.py <config>.yaml
conda run -n SynThera_ExptDesigns python executables/runDORAXGB.py <config>.yaml
```

Results are written to the same folder as the config. The `*_config.yaml` files in `executables/` are templates, so copy one into your run folder before editing.

The image already includes RetroTide, DORAnet and DORA-XGB. A DORAnet checkout sits at `/opt/synthera-deps/doranet`, and `doranet_config.yaml` points to it. To pin a tool to a commit, see [`build/DOCKER.md`](build/DOCKER.md#bundled-tools-versus-art).

## Known limitations

- `executables/admet_config.yaml` uses the placeholder `<PROJECT_ROOT>` for `inputFile` and `outputDir`. Replace it with your own paths before use.
- `external/ergochemics` is added to `sys.path` and not installed. It declares `python>=3.12` and `rdkit>=2024.9.5`, but `SynThera_ExptDesigns` has Python 3.10.19 and rdkit 2023.09.5. The Step5 notebooks patch one resulting API gap (`Chem.MolToSmiles`). Others may exist.
- The pinned packages in `environment.yml` are for CUDA and have been tested only on Linux x86-64.

## Manual setup without Docker

You need conda (Miniconda or Miniforge; Mamba is used if present) and a Linux x86-64 host, because both specs are exact linux-64 exports. On macOS or Windows, use Docker.

```bash
cd build
python prepareEnvSpecs.py   # writes copies of the specs that install from PyPI
python setup.py             # creates both environments and registers kernels
```

`prepareEnvSpecs.py` edits the specs in place, so back up `build/environment.yml` first. It removes the `art`, `pathermo`, `doranet` and `dora-xgb` pip entries, which do not resolve from PyPI, and renames `python-graphviz` to `graphviz`. `setup.py` accepts `--force`, `--skip-kernels` and `--only <env>`.

Then install the three tools that the Docker image adds:

```bash
conda run -n SynThera_ExptDesigns pip install --no-deps mapchiral
conda run -n SynThera_ExptDesigns pip install --no-deps "git+https://github.com/JBEI/RetroTide.git"

git clone https://github.com/wsprague-nu/doranet.git ~/doranet
conda run -n SynThera             pip install --no-deps ~/doranet
conda run -n SynThera_ExptDesigns pip install --no-deps ~/doranet
# set doranetPath in executables/doranet_config.yaml to ~/doranet

conda run -n SynThera_ExptDesigns pip install --no-deps "git+https://github.com/tyo-nu/DORA_XGB.git"
```

## Troubleshooting

- `import art` fails: ART is not installed. See [Installing ART](#installing-art-required-for-step2-and-step4).
- A script cannot find its input: check the paths in its config YAML (see [Known limitations](#known-limitations)).
- The kernel is missing in Jupyter (manual setup): rerun `python setup.py --only <env>`.
- A pip package fails to build (manual setup): install system build tools such as `build-essential`, or use Docker.
