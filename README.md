# MolScribeServerAPI

A small, standalone HTTP server that turns images of chemical structures into SMILES using
[MolScribe](https://github.com/thomas0809/MolScribe). It is a companion to
[DecimerServerAPI](https://github.com/DocMinus/DecimerServerAPI) and speaks the **same HTTP
contract**, so a DECIMER client can talk to it with almost no changes.

Why a server? MolScribe pins `numpy<2` and (via its metadata) `torch<2.0`, and its internal
`multiprocessing.Pool` misbehaves under macOS "spawn". Wrapping it in a Linux container keeps
those constraints isolated — the calling app stays on
modern numpy/torch and just makes HTTP calls, and the whole thing runs even on a Mac host.

## Endpoints

See [API_ENDPOINTS.md](API_ENDPOINTS.md). In short:

- `GET /` — health check
- `GET /system/status` — accelerator (`cpu`/`cuda`/`mps`) + torch version
- `POST /image2smiles/` — `{"encoded_image": "<base64>"}` → `{"smiles", "molfile", "confidence", "reason"}`

## Run it (CPU)

```shell
docker compose up -d --build         # builds the CPU image (linux/amd64) and starts it on port 8999
curl http://localhost:8999/          # -> {"Message": "...up and running."}
```

This is a **linux/amd64** image (the CPU-only torch wheels are x86_64; it runs on Apple Silicon
via emulation). torch is pinned to the CPU-only build, so the image is small (~1.5–2 GB) instead
of the ~9 GB the default CUDA torch wheel would drag in. For GPU, use the CUDA image below.

The first `POST /image2smiles/` downloads the MolScribe checkpoint (~400 MB) into the
`molscribeserverapi_model_files` volume (mounted at `/home/appuser/.data/molscribe`), so it
persists across restarts. Subsequent conversions take a few seconds on CPU.

Quick test with a base64 PNG:

```shell
curl -s -X POST http://localhost:8999/image2smiles/ \
  -F "encoded_image=$(base64 -i yourimage.png)"
```

## Run it (CUDA GPU)

`Dockerfile.cuda` installs the **stock** PyTorch wheel, which is the CUDA build (`torch==2.13.0`
ships as `+cu130`, bundling the CUDA 13 userspace libraries). The GPU *driver* comes from the
host at run time, so this needs an NVIDIA host with:

- a driver new enough for CUDA 13 (>= ~580; e.g. driver `595.71.x`),
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) installed.

```shell
docker compose -f docker-compose.cuda.yml up -d --build   # grants the GPU via a device reservation
curl http://localhost:8999/system/status                  # -> {"accelerator_type": "cuda", ...}
```

`molscribe_server.py` auto-selects the GPU when `torch.cuda.is_available()`, so no code change is
needed. Tested on Linux box with CUDA 13 / driver 595.71.x. 
If the container reports `accelerator_type: cpu`, the GPU wasn't granted — check
the driver version and that the toolkit/`docker-compose.cuda.yml` device reservation are in effect.

## Local development (without Docker)

```shell
uv sync                              # provision the environment from uv.lock
uv pip install --no-deps molscribe   # MolScribe (its torch<2.0 pin is intentionally bypassed)
uv run python molscribe_server.py    # serves on http://localhost:8999
```

## Packaging notes

- Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`). MolScribe is **not** a
  locked dependency: its metadata caps `torch<2.0`, so it is installed separately with
  `--no-deps` while a modern torch (and MolScribe's other real deps) come from the lock. Only
  the packages MolScribe actually imports are included — `transformers`/`matplotlib`/
  `tensorboardX` from the original eval recipe are dropped because they're never imported.
- **CPU vs CUDA torch.** The CPU image pins torch/torchvision to PyTorch's CPU-only index
  (`[tool.uv.sources]` in `pyproject.toml`) so it doesn't drag in the ~3.5 GB of NVIDIA CUDA
  libraries the default wheel bundles. The CUDA image (`Dockerfile.cuda`) deliberately installs
  the stock (CUDA) wheel instead. The two images are specialized on purpose: small CPU, GPU-capable CUDA.
- Image tag: `docminus/molscribe_api` (CPU, linux/amd64). Port `8999`, volume
  `molscribeserverapi_model_files`, network `molscribenetwork`. Build/push helpers are in the
  [Makefile](Makefile) (`make buildx` local; `make release VERSION=x.y.z` push).

## License

MIT — see [LICENSE](LICENSE). MolScribe and its model weights are under their own license;
please cite Qian et al., *J. Chem. Inf. Model.* 2023, https://doi.org/10.1021/acs.jcim.2c01480.
