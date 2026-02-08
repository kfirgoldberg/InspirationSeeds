# /// script
# dependencies = [
#   "diffusers @ https://github.com/huggingface/diffusers.git",
#   "torch",
#   "transformers",
#   "peft",
#   "accelerate",
#   "protobuf",
#   "sentencepiece",
#   "huggingface_hub",
# ]
# ///


import argparse
import tempfile
import urllib.request
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from PIL import Image

DEFAULT_PROMPT = (
    "combine the element in the top left with the element in the bottom right to create a single object inspired by "
    "both of them"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLUX Kontext with LoRA to combine two images into a single inspired object."
    )
    parser.add_argument("--image1", type=Path, required=True, help="Path to the first input image (top-left).")
    parser.add_argument("--image2", type=Path, required=True, help="Path to the second input image (bottom-right).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory where outputs will be written.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Guidance scale to use for generation (higher = closer adherence to prompt).",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="LoRA checkpoint: local path, HuggingFace repo (e.g., 'user/repo'), or URL.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed for generation.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=4,
        help="Number of seeds to generate (starting from --seed and incrementing). Default: 4.",
    )
    return parser.parse_args()




def load_lora_weights(pipe: FluxKontextPipeline, lora_path: str) -> None:
    """Load LoRA weights from a local path, HuggingFace repo, or URL."""
    path = Path(lora_path)

    if path.exists():
        # Local file path
        print(f"Loading LoRA from local path: {lora_path}")
        pipe.load_lora_weights(str(path), adapter_name="default")
    elif lora_path.startswith(("http://", "https://")):
        # URL - download to temp file
        print(f"Downloading LoRA from URL: {lora_path}")
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            urllib.request.urlretrieve(lora_path, tmp.name)
            pipe.load_lora_weights(tmp.name, adapter_name="default")
    else:
        # Assume HuggingFace repo format (e.g., "user/repo" or "user/repo/file.safetensors")
        print(f"Loading LoRA from HuggingFace: {lora_path}")
        parts = lora_path.split("/")
        if len(parts) >= 3:
            # Format: user/repo/path/to/file.safetensors
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            pipe.load_lora_weights(local_path, adapter_name="default")
        else:
            # Format: user/repo - let diffusers handle it directly
            pipe.load_lora_weights(lora_path, adapter_name="default")


def create_combined_image(left: Image.Image, right: Image.Image) -> Image.Image:
    """Create a side-by-side combined image."""
    left_rgb = left.convert("RGB")
    right_rgb = right.convert("RGB")
    total_width = left_rgb.width + right_rgb.width
    max_height = max(left_rgb.height, right_rgb.height)

    combined = Image.new("RGB", (total_width, max_height))
    combined.paste(left_rgb, (0, 0))
    combined.paste(right_rgb, (left_rgb.width, 0))
    return combined


def create_input_grid(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """Create a 1024x1024 grid input for Flux Kontext from two images.

    Both images are resized to 512x512 and placed on a 1024x1024 white canvas.
    img1 is placed in top-left, img2 in bottom-right.
    """
    img1_resized = img1.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    img2_resized = img2.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    grid = Image.new("RGB", (1024, 1024), "white")
    grid.paste(img1_resized, (0, 0))
    grid.paste(img2_resized, (512, 512))

    return grid


def main() -> None:
    args = parse_args()

    image1_path: Path = args.image1.expanduser()
    image2_path: Path = args.image2.expanduser()
    output_dir: Path = args.output_dir.expanduser()

    if not image1_path.is_file():
        raise ValueError(f"Image1 '{image1_path}' does not exist or is not a file.")
    if not image2_path.is_file():
        raise ValueError(f"Image2 '{image2_path}' does not exist or is not a file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input images and create grid
    print(f"Loading images: {image1_path.name} + {image2_path.name}")
    img1 = load_image(str(image1_path))
    img2 = load_image(str(image2_path))
    input_grid = create_input_grid(img1, img2)

    # Initialize pipeline
    print("Loading FLUX Kontext pipeline...")
    pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)

    if args.lora_path:
        load_lora_weights(pipe, args.lora_path)

    pipe.to("cuda")
    device = pipe.device if hasattr(pipe, "device") else torch.device("cuda")

    # Generate output filename prefix from input image names
    output_prefix = f"{image1_path.stem}_x_{image2_path.stem}"

    print(f"Generating {args.num_seeds} images with seeds {args.seed} to {args.seed + args.num_seeds - 1}")

    for i in range(args.num_seeds):
        current_seed = args.seed + i
        output_filename = f"{output_prefix}__seed_{current_seed:03d}.jpeg"
        output_path = output_dir / output_filename

        if output_path.exists():
            print(f"[{i + 1}/{args.num_seeds}] Skipping (exists): {output_filename}")
            continue

        print(f"[{i + 1}/{args.num_seeds}] Generating with seed {current_seed}...")

        generator = torch.Generator(device=device).manual_seed(current_seed)
        result_image = pipe(
            image=input_grid,
            prompt=DEFAULT_PROMPT,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]

        result_image.save(output_path)

        # Save combined image (input grid + result)
        combined_image = create_combined_image(input_grid, result_image)
        combined_output_path = output_dir / f"{output_prefix}__seed_{current_seed:03d}_combined.jpeg"
        combined_image.save(combined_output_path, format="JPEG")

        print(f"[{i + 1}/{args.num_seeds}] Saved: {output_filename}")

    print("Done!")


if __name__ == "__main__":
    main()
