# ComfyUI-MOSS-Image-Ostris-Edit

Nodes for running MOSS-Image edit LoRAs trained with ai-toolkit (the
`MOSS-Image Base [Edit Training]` / `MOSS-Image SFT [Edit Training]` models,
i.e. the `moss_image` arch with `model_kwargs.edit: true`).

## Installation

Clone this repo into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit.git
```

No extra dependencies are required. The nodes appear under the
`ostris/moss_image` category. Needs a ComfyUI with native MOSS-Image support
(`CLIPLoader` type `moss_image`).

## Nodes

### Text Encode MOSS-Image Ostris Edit

Inputs: `clip`, `prompt`, optional `vae`, `image1`, `mask1`, `image2`, `image3`.
Output: `CONDITIONING`.

Encodes the prompt together with the reference images through the MOSS-Image
Qwen3.5 text encoder, using MOSS's chat template with `Picture N:` vision
placeholders ahead of the prompt — the same layout used during training.
When a VAE is connected, each reference image is also VAE-encoded and attached
to the conditioning as reference latents for the model patch node.

Images fed to the Qwen3.5 encoder are downscaled (never upscaled) to fit
384x384 total pixels, as in training. Reference latents are encoded at the
resolution the image is fed in (only snapped to the FLUX.2 VAE's 16 px grid):
**scale the reference yourself before this node** — the example workflow
runs it through `FluxKontextImageScale` (about 1MP, the training default).
A 1024x1024 reference adds 4096 tokens to every step; a 512x512 one 1024.

`mask1` places image1 in the output. Reference and output share a centered
coordinate grid, so a reference the size of the output lines up with it
pixel for pixel. Pad image1 to the output size with the subject where you want
it, and mask the padding (1 = masked, e.g. the transparent area from Load
Image's `MASK`). Masked 16 px latent cells are dropped from the reference, the
kept ones keep their positions, and the model generates the masked area. The
Qwen3.5 encoder sees image1 cropped to the unmasked area.

Note: the text encoder checkpoint must include the Qwen3.5 vision weights
(the Comfy-Org `qwen3.5_9b_*` files do) or the images cannot be encoded.

### MOSS-Image Ostris Edit Model Patch

Inputs: `model`, `kv_cache` (default on). Output: `model`.

Patches the MOSS-Image model so it consumes the reference latents from the
conditioning. Each reference is appended to the token sequence after the
text, on its own index of the RoPE axis the base model leaves at 0, and
conditioned at timestep 0 (the `index_timestep_zero` reference method); the
denoising prediction covers only the target image tokens. This node is
required because the stock MOSS-Image model in ComfyUI ignores reference
latents.

If the conditioning has no reference latents, the patched model behaves
exactly like the stock model, so it is safe to leave in the graph.

`kv_cache` caches the reference tokens' attention K/V: they are precomputed in
a single t=0 pass and reused on every denoising step, so the references never
ride along in the per-step sequence (about 2x faster than carrying them). It
is **on by default** because ai-toolkit's MOSS-Image edit models train with
`kv_cache` on. **Turn it off for a LoRA trained with `kv_cache: false`**, where
the reference tokens were allowed to attend to the text and noisy tokens.

Layer skipping (MOSS's default: block 22 skipped on the negative branch with
the `SkipLayerGuidanceDiTSimple` node, `single_layers: 22`) works with both
modes, including the cached reference pass.

## Example wiring

```
Load Diffusion Model (moss_image) -> Load LoRA -> MOSS-Image Ostris Edit Model Patch
    -> SkipLayerGuidanceDiTSimple (single_layers 22) -> KSampler (30 steps, cfg 3)
CLIPLoader (moss_image) -> Text Encode MOSS-Image Ostris Edit (prompt + images + VAE) -> positive
CLIPLoader (moss_image) -> Text Encode MOSS-Image Ostris Edit (negative prompt + same images + VAE) -> negative
```

An example workflow is in `workflow/MOSS_Image_Ostris_Edit.json`.
