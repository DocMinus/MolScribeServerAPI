# API Endpoints

HTTP endpoints exposed by `molscribe_server.py`.

Base URL (default local): `http://localhost:8999`

The contract mirrors the DECIMER server (DocMinus/DecimerServerAPI) so an existing DECIMER
client can point at this server with minimal changes. Differences from DECIMER: there is no
image classifier and no `is_hand_drawn` / `classify_image` inputs; the success response adds a
`confidence` and a `molfile`, and `/system/status` reports `torch_version` instead of
`tensorflow_version`.

## `GET /`

Simple health/status endpoint.

### Response `200`

```json
{
  "Message": "Image2smiles (MolScribe) converter is up and running."
}
```

## `GET /system/status`

Check hardware acceleration availability and system information at runtime.

### Response `200`

```json
{
  "status": "ready",
  "accelerator_type": "cpu",
  "torch_version": "2.13.0+cpu"
}
```

The CPU image reports `accelerator_type: cpu` and a `+cpu` torch build; the CUDA image
(`Dockerfile.cuda`) reports `cuda` and a CUDA torch build (e.g. `2.13.0+cu130`) when a GPU is
granted.

Possible `accelerator_type` values:
- `cuda`: NVIDIA CUDA GPU (the `Dockerfile.cuda` image on a GPU host)
- `mps`: Apple Metal (only when run outside Docker on Apple Silicon)
- `cpu`: CPU-only processing (the default CPU image)

## `POST /image2smiles/`

Convert a base64-encoded image into a SMILES string.

### Request formats

Supported content types:
- `application/x-www-form-urlencoded`
- `multipart/form-data`
- `application/json`

Fields:
- `encoded_image` (required, string): base64-encoded image bytes (PNG/JPEG)

### Successful response `200`

```json
{
  "smiles": "Fc1ccc(-c2cc(-c3ccccc3)n(-c3ccccc3)c2)cc1",
  "molfile": "\n     RDKit          2D\n\n ...",
  "confidence": 0.905,
  "reason": null
}
```

`confidence` is MolScribe's own `overall_score` for the prediction (roughly
average-token-score × edge-score-product). It reflects how self-consistent the prediction is,
**not** whether the image is a molecule — noise/blank images still return a (low-confidence)
SMILES or `no_structure`, so do not treat it as a "not a molecule" classifier.

When no structure could be produced, `smiles` (and `molfile`, `confidence`) are `null` and
`reason` explains why:

```json
{
  "smiles": null,
  "molfile": null,
  "confidence": null,
  "reason": "no_structure"
}
```

Possible `reason` values when `smiles` is `null`:
- `no_structure` — the model returned an empty SMILES
- `decode_failed` — the bytes weren't a decodable image
- `prediction_failed` — MolScribe raised while processing the image

### Validation and error responses

- `400` malformed request payload (invalid JSON/form)
- `400` invalid base64 in `encoded_image`
- `413` `encoded_image` payload too large
- `422` missing/empty/invalid field values (for example missing `encoded_image`)

Error body format:

```json
{
  "message": "<error detail>"
}
```

## Environment Variables

- `MOLSCRIBE_MAX_ENCODED_IMAGE_BYTES` (int, default `6291456`): max encoded payload size in bytes
- `HF_HOME` (set to `/home/appuser/.data/molscribe` in the images): HuggingFace cache dir; the
  ~400 MB checkpoint downloads here on first request, so mount it as a volume to persist it
