"""
Mini-server for MolScribe image-to-SMILES conversion.

Mirrors the DECIMER server (DocMinus/DecimerServerAPI) so chemdash can talk to it
with the exact same HTTP contract, but backed by MolScribe
(https://github.com/thomas0809/MolScribe, https://doi.org/10.1021/acs.jcim.2c01480)
instead of DECIMER.

Design notes:
  * The model is loaded once at import (module global), on CUDA when available else CPU.
  * MolScribe's graph->SMILES step spins up ``multiprocessing.Pool(16)`` to parallelise a
    *batch*; we convert one image per request, so those workers are pure overhead. We swap
    the Pool for a synchronous shim (see ``_SyncPool``). On Linux (fork) this just avoids the
    overhead; it also makes the server robust to any start method.
  * MolScribe itself is installed with ``--no-deps`` (its metadata pins ``torch<2.0``); its
    real runtime deps are provided by pyproject.toml with a modern torch.

license: MIT
"""

import base64
import binascii
import os
import types
from io import BytesIO

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOLSCRIBE_REPO = "yujieq/MolScribe"
MOLSCRIBE_CHECKPOINT = "swin_base_char_aux_1m.pth"


def _get_max_payload_size() -> int:
    """Returns the max allowed encoded payload size in bytes (default 6 MB)."""

    default_limit = 6 * 1024 * 1024
    configured_limit = os.getenv("MOLSCRIBE_MAX_ENCODED_IMAGE_BYTES")
    if configured_limit is None:
        return default_limit
    try:
        limit = int(configured_limit)
    except ValueError:
        return default_limit
    return limit if limit > 0 else default_limit


MAX_ENCODED_IMAGE_BYTES: int = _get_max_payload_size()


def _get_accelerator_type() -> str:
    """Detects the available hardware accelerator type.

    Returns one of ``'cuda'``, ``'mps'`` or ``'cpu'``.
    """

    try:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Model (loaded once at import)
# ---------------------------------------------------------------------------


class _SyncPool:
    """Drop-in for ``multiprocessing.Pool`` that runs everything in-process.

    MolScribe's ``convert_graph_to_smiles`` calls ``multiprocessing.Pool(16).starmap(...)``
    to parallelise a batch. We convert one image at a time, so the 16 workers are pure
    overhead; running the (RDKit) conversion synchronously sidesteps it and is robust to the
    process start method.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starmap(self, func, iterable, chunksize=None):
        return [func(*args) for args in iterable]

    def map(self, func, iterable, chunksize=None):
        return [func(x) for x in iterable]


def _load_model():
    """Download the checkpoint (cached) and load MolScribe onto the best device."""

    from molscribe import MolScribe
    from molscribe import chemistry as _chem

    # neutralise MolScribe's internal multiprocessing.Pool (see _SyncPool)
    _chem.multiprocessing = types.SimpleNamespace(Pool=_SyncPool)

    ckpt_path = hf_hub_download(MOLSCRIBE_REPO, MOLSCRIBE_CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MolScribe(ckpt_path, device=device)


MODEL = _load_model()


def _predict_smiles(decoded_bytes: bytes) -> dict:
    """Convert raw image bytes to a MolScribe prediction dict.

    Returns ``{"smiles", "molfile", "confidence"}`` (values may be ``None``) or raises
    ``_DecodeError`` / ``_PredictionError`` which the endpoint maps to a ``reason``.
    """

    try:
        image = Image.open(BytesIO(decoded_bytes))
        rgb = np.asarray(image.convert("RGB"))
    except Exception:
        raise _DecodeError()

    try:
        result = MODEL.predict_image(rgb, return_confidence=True)
    except Exception:
        raise _PredictionError()

    confidence = result.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

    smiles = result.get("smiles") or None  # normalise empty string to None
    return {"smiles": smiles, "molfile": result.get("molfile"), "confidence": confidence}


class _DecodeError(Exception):
    """The payload decoded to bytes but wasn't a usable image."""


class _PredictionError(Exception):
    """MolScribe raised while processing an otherwise-valid image."""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()


async def _extract_request_params(request: Request) -> bytes:
    """Accept a form or JSON body; validate and return the decoded image bytes.

    Mirrors the DECIMER server: ``encoded_image`` (base64 string) is required, non-empty,
    within ``MAX_ENCODED_IMAGE_BYTES`` and valid base64.
    """

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed JSON payload.")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON payload must be an object.")
    else:
        try:
            payload = dict(await request.form())
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed form payload.")

    encoded_image = payload.get("encoded_image")
    if encoded_image is None:
        raise HTTPException(status_code=422, detail="Field 'encoded_image' is required.")
    if not isinstance(encoded_image, str):
        raise HTTPException(status_code=422, detail="Field 'encoded_image' must be a base64 string.")

    encoded_image = encoded_image.strip()
    if not encoded_image:
        raise HTTPException(status_code=422, detail="Field 'encoded_image' must not be empty.")

    if len(encoded_image.encode("utf-8")) > MAX_ENCODED_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Payload too large for 'encoded_image'. " f"Maximum allowed size is {MAX_ENCODED_IMAGE_BYTES} bytes."
            ),
        )

    try:
        decoded_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Field 'encoded_image' is not valid base64.")

    if not decoded_bytes:
        raise HTTPException(status_code=422, detail="Field 'encoded_image' decodes to empty content.")

    return decoded_bytes


@app.get("/")
async def root():
    return {"Message": "Image2smiles (MolScribe) converter is up and running."}


@app.get("/system/status")
async def system_status():
    """Returns system status including hardware accelerator type and torch version."""
    return {
        "status": "ready",
        "accelerator_type": _get_accelerator_type(),
        "torch_version": torch.__version__,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catches general (e.g. Python) exceptions."""
    return JSONResponse(
        status_code=500,
        content={"message": "An unspecified general error in image2smiles API occurred."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Catches HTTP exceptions."""
    return JSONResponse(status_code=exc.status_code, content={"message": str(exc.detail)})


@app.post("/image2smiles/")
async def image_to_smiles(request: Request):
    """Convert a base64-encoded image to a SMILES string via MolScribe.

    encoded_image: base64 encoded image (as string)
    """

    decoded_bytes = await _extract_request_params(request)

    try:
        prediction = _predict_smiles(decoded_bytes)
    except _DecodeError:
        return {"smiles": None, "molfile": None, "confidence": None, "reason": "decode_failed"}
    except _PredictionError:
        return {"smiles": None, "molfile": None, "confidence": None, "reason": "prediction_failed"}

    if prediction["smiles"] is None:
        return {"smiles": None, "molfile": None, "confidence": None, "reason": "no_structure"}

    return {
        "smiles": prediction["smiles"],
        "molfile": prediction["molfile"],
        "confidence": prediction["confidence"],
        "reason": None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8999)
