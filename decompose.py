# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "torch",
#     "diffusers",
#     "scikit-learn",
#     "tqdm",
#     "transformers",
#     "huggingface-hub",
# ]
# ///

import importlib.util
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.utils import load_image
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

DEVICE = "cuda"


def cluster_clip_embeddings(
    clip_embeds_dict: Dict[int, np.ndarray],
    k: int = 2,
    random_state: int | None = 0,
    **kmeans_kwargs,
) -> Dict[int, List[int]]:
    """
    Cluster CLIP-style embeddings with K-means and group the original keys by cluster.

    Parameters
    ----------
    clip_embeds_dict : dict[int, np.ndarray]
        Maps an integer index to a (1280,) embedding vector.
    k : int, default 2
        Number of clusters.
    random_state : int or None, default 0
        Seed for KMeans reproducibility.  Use None for stochastic runs.
    **kmeans_kwargs
        Extra keyword arguments forwarded to sklearn.cluster.KMeans.

    Returns
    -------
    dict[int, list[int]]
        Keys are cluster labels (0 … k-1); values are the integer indices
        whose embeddings fell into each cluster.
    """
    # 1. Unpack keys and stack embeddings into a 2-D array
    keys = list(clip_embeds_dict)  # preserves insertion order
    X = np.stack([clip_embeds_dict[k] for k in keys])  # shape = (n_samples, 1280)

    # 2. Run K-means
    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init="auto",  # or an int for explicit restarts if using sklearn < 1.4
        **kmeans_kwargs,
    )
    labels = km.fit_predict(X)  # array shape = (n_samples,)

    # 3. Group keys by their assigned label
    clusters: dict[int, list[int]] = defaultdict(list)
    for key, label in zip(keys, labels, strict=True):
        clusters[int(label)].append(key)

    return dict(clusters), km.cluster_centers_


@torch.inference_mode()
def embed_image(item, prior):
    return prior.interpolate([item], [1]).image_embeds[0]


@torch.inference_mode()
def generate_one(image_embeds, decoder, negative_emb):
    generator = torch.Generator("cuda")
    generator.manual_seed(1)
    images = decoder(
        image_embeds=image_embeds.unsqueeze(0),
        negative_image_embeds=negative_emb,
        height=512,
        width=512,
        num_inference_steps=18,
        num_images_per_prompt=1,
        generator=generator,
    ).images

    return images[0]


def clean_clusters_by_distance(
    clusters_dict: Dict[str, Dict[str, Dict[str, float]]],
    cluster_centers: np.ndarray,
    mean_clip_embeds_dict: Dict[int, np.ndarray],
    removal_fraction: float = 0.20,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Clean clusters by removing features that are closest to the opposing cluster center.

    For each cluster, removes the specified fraction of features that are closest
    to the other cluster's center, helping to create more distinct clusters.

    Parameters
    ----------
    clusters_dict : dict[str, dict[str, dict[str, float]]]
        Nested dict with cluster structure: {"0": {"feature_id": {"feature_id": id, "feature_value": val}}}
    cluster_centers : np.ndarray
        Array of cluster centers with shape (n_clusters, n_features).
    mean_clip_embeds_dict : dict[int, np.ndarray]
        Maps feature index to its embedding vector.
    removal_fraction : float, default 0.20
        Fraction of features to remove from each cluster (those closest to other center).

    Returns
    -------
    tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, dict[str, float]]]]
        A tuple containing:
        - clean_clusters: Dict preserving original nested structure with filtered features
        - dropped_features: Dict preserving original nested structure with dropped features
    """
    if len(clusters_dict) != 2:
        raise ValueError(f"This function is designed for 2 clusters, but found {len(clusters_dict)}.")

    clean_clusters = {}
    dropped_features = {}

    cluster_labels = sorted(list(clusters_dict.keys()))

    for label in cluster_labels:
        # Get the other label (assumes binary clustering)
        other_label = cluster_labels[1 - cluster_labels.index(label)]
        label_int = int(label)
        other_label_int = int(other_label)

        # Extract feature indices and their data from nested structure
        current_cluster = clusters_dict[label]
        current_feature_indices = [int(feat_dict["feature_id"]) for feat_dict in current_cluster.values()]
        current_embeddings = np.stack([mean_clip_embeds_dict[idx] for idx in current_feature_indices])

        other_center = cluster_centers[other_label_int]

        # Calculate distances to the other cluster center
        distances_other = np.linalg.norm(current_embeddings - other_center, axis=1)

        # Create list of (feature_idx, distance, original_key, original_data) and sort by distance
        feature_distance_data = []
        for i, (feat_key, feat_data) in enumerate(current_cluster.items()):
            feature_distance_data.append((current_feature_indices[i], distances_other[i], feat_key, feat_data))

        feature_distance_data.sort(key=lambda x: x[1])  # Sort by distance, closest first

        # Remove the closest features (bottom fraction)
        num_to_remove = int(len(feature_distance_data) * removal_fraction)
        features_to_remove_data = feature_distance_data[:num_to_remove]
        features_to_keep_data = feature_distance_data[num_to_remove:]

        # Rebuild nested structure for clean clusters
        clean_clusters[label] = {}
        for _, _, feat_key, feat_data in features_to_keep_data:
            clean_clusters[label][feat_key] = feat_data

        # Rebuild nested structure for dropped features
        dropped_features[label] = {}
        for _, _, feat_key, feat_data in features_to_remove_data:
            dropped_features[label][feat_key] = feat_data

    return clean_clusters, dropped_features


@torch.inference_mode()
def load_sae():
    from huggingface_hub import snapshot_download

    model_file_path = snapshot_download(repo_id="gytdau/clip-sae-128")

    model_py_path = os.path.join(model_file_path, "model.py")
    weights_path = os.path.join(model_file_path, "sparse_autoencoder_128.pth")

    spec = importlib.util.spec_from_file_location("model_module", model_py_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    SparseAutoencoder = getattr(model_module, "SparseAutoencoder")

    # Create an instance of the model
    embedding_dim = 1280
    hidden_dim = embedding_dim * 128
    model = SparseAutoencoder(embedding_dim, hidden_dim)

    # Load the state dict
    state_dict = torch.load(weights_path, map_location=torch.device("cuda"))
    model.half()
    model.load_state_dict(state_dict)
    model.eval()
    model.to("cuda")

    return model


@torch.inference_mode()
def load_kandinsky():
    image_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", subfolder="image_encoder"
        )
        .half()
        .to(DEVICE)
    )

    unet = (
        UNet2DConditionModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="unet")
        .half()
        .to(DEVICE)
    )

    prior = KandinskyV22PriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior", image_encoder=image_encoder, torch_dtype=torch.float16
    ).to(DEVICE)

    decoder = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", unet=unet, torch_dtype=torch.float16
    ).to(DEVICE)

    zero_embed = prior.get_zero_embed()

    return prior, decoder, zero_embed


def cluster_sae_by_images_from_web(sae_model, top_k_idxs: List[int], sae_features):
    clip_embeds_dict = {}
    for idx in top_k_idxs:
        idx = idx.item()
        one_hot_vec = torch.zeros_like(sae_features)
        one_hot_vec[idx] = 1
        clip_embeds_dict[idx] = sae_model.decode(one_hot_vec).cpu().detach().float().numpy()

    mean_clip_embeds_dict = clip_embeds_dict
    clusters, cluster_centers = cluster_clip_embeddings(mean_clip_embeds_dict, k=2)

    clusters_json = {}
    for k, v in clusters.items():
        clusters_json[k] = {}
        for idx in v:
            feat_dict = {"feature_id": idx, "feature_value": sae_features[idx].item()}
            clusters_json[k][idx] = feat_dict

    return clusters_json, mean_clip_embeds_dict, cluster_centers


@torch.inference_mode()
def run(
    output_dir: Path,
    prior,
    decoder,
    negative_emb,
    model,
    top_k: int = 32,
    apply_clean_clusters_frac: float = 0.0,
    base_img_url: str = None,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    base_image = load_image(base_img_url)
    base_image_clip_embed = embed_image(item=base_image, prior=prior)

    base_image_sae_features = model.encode(base_image_clip_embed)
    top_k_idxs = base_image_sae_features.argsort()[-top_k:]
    top_k_idxs = reversed(top_k_idxs)
    clusters_json, mean_clip_embeds_dict, cluster_centers = cluster_sae_by_images_from_web(
        sae_model=model,
        top_k_idxs=top_k_idxs,
        sae_features=base_image_sae_features,
    )
    if apply_clean_clusters_frac > 0:
        clusters_json, _ = clean_clusters_by_distance(
            clusters_dict=clusters_json,
            cluster_centers=cluster_centers,
            mean_clip_embeds_dict=mean_clip_embeds_dict,
            removal_fraction=apply_clean_clusters_frac,
        )
    cluster_to_sae_vec_dict = {}
    for cluster_id, cluster_members in clusters_json.items():
        cluster_sae_vec = torch.zeros_like(base_image_sae_features)
        for member_data in cluster_members.values():
            if "feature_id" in member_data and "feature_value" in member_data:
                feature_id = member_data["feature_id"]
                feature_value = member_data["feature_value"]
                cluster_sae_vec[feature_id] = feature_value
        cluster_to_sae_vec_dict[cluster_id] = cluster_sae_vec

    cluster1_sae_vec = cluster_to_sae_vec_dict[1]
    cluster0_sae_vec = cluster_to_sae_vec_dict[0]
    ref_img = load_image(base_img_url)
    ref_img_clip_embed = embed_image(item=ref_img, prior=prior)

    direction_sae_vec = cluster1_sae_vec - cluster0_sae_vec
    direction_clip_vec = model.decoder(direction_sae_vec)
    step_sizes = [-1, -0.5, 0, 0.5, 1]

    for step_size in tqdm(step_sizes):
        edited_clip_features = ref_img_clip_embed.clone()
        edited_clip_features = edited_clip_features + step_size * direction_clip_vec
        output_path = output_dir / f"step_{step_size}.jpeg"
        generated_image = generate_one(
            image_embeds=edited_clip_features, decoder=decoder, negative_emb=negative_emb
        )
        generated_image.save(output_path)


if __name__ == "__main__":
    # base_img_name = "mug2"
    import argparse

    def parse_int_list(value: str) -> list[int]:
        return [int(x.strip()) for x in value.split(",")]

    def parse_float_list(value: str) -> list[float]:
        return [float(x.strip()) for x in value.split(",")]

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Decompose an image into clustered SAE features and generate edited variations"
        )
        parser.add_argument(
            "--image",
            type=str,
            required=True,
            help="Path to the input image file",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default="outputs",
            help="Output directory for results (default: outputs)",
        )
        parser.add_argument(
            "--top-k",
            type=parse_int_list,
            default=[32, 64],
            help="Comma-separated list of top-k values (default: 32,64)",
        )
        parser.add_argument(
            "--clean-clusters-frac",
            type=parse_float_list,
            default=[0.5, 0.7],
            help="Comma-separated list of cluster cleaning fractions (default: 0.5,0.7)",
        )
        return parser.parse_args()

    from itertools import product

    args = parse_args()
    prior, decoder, negative_emb = load_kandinsky()
    model = load_sae()

    img_name = Path(args.image).stem
    combinations = list(product(args.top_k, args.clean_clusters_frac))
    print(f"Running {len(combinations)} combinations: {combinations}")

    for top_k, clean_frac in combinations:
        output_dir = Path(args.output_dir) / f"{img_name}_topk_{top_k}_frac_{clean_frac}"
        print(f"Running top_k={top_k}, clean_frac={clean_frac} -> {output_dir}")

        run(
            output_dir=output_dir,
            base_img_url=args.image,
            top_k=top_k,
            apply_clean_clusters_frac=clean_frac,
            prior=prior,
            decoder=decoder,
            negative_emb=negative_emb,
            model=model,
        )
