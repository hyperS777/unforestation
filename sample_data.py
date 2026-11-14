"""
High-Fidelity Sample Dataset Generator for Western Ghats Study Districts:
Pune, Satara, and Sindhudurg.
Enables immediate out-of-the-box exploration, testing, and interactive visualization
without requiring manual uploads or external Google Drive mounting.
"""

import os
import numpy as np
from rasterio.transform import from_bounds
import rasterio

DISTRICT_AOIS = {
    "Pune": {
        "bounds": (73.55, 18.35, 73.58, 18.38),
        "center": [18.365, 73.565],
        "forest_density": 0.45,
        "loss_rate": 0.038,
        "gain_rate": 0.012,
        "description": "Northern Western Ghats - Mixed moist deciduous & agro-forestry fringe"
    },
    "Satara": {
        "bounds": (73.75, 17.70, 73.78, 17.73),
        "center": [17.715, 73.765],
        "forest_density": 0.62,
        "loss_rate": 0.024,
        "gain_rate": 0.018,
        "description": "Central Western Ghats - Dense semi-evergreen ridge & plateau forest"
    },
    "Sindhudurg": {
        "bounds": (73.65, 15.95, 73.68, 15.98),
        "center": [15.965, 73.665],
        "forest_density": 0.74,
        "loss_rate": 0.019,
        "gain_rate": 0.022,
        "description": "Southern Maharashtra Sahyadri - High-canopy wet evergreen corridors"
    }
}


def generate_synthetic_district_data(district_name="Pune", height=330, width=330, seed=42):
    """
    Generates realistic 4-band Sentinel-2 reflectance, ESA forest masks,
    model probabilities, and change maps for a district.
    """
    cfg = DISTRICT_AOIS.get(district_name, DISTRICT_AOIS["Pune"])
    rng = np.random.default_rng(seed + hash(district_name) % 1000)

    bounds = cfg["bounds"]
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    meta = {
        "crs": "EPSG:4326",
        "transform": transform,
        "width": width,
        "height": height,
        "bounds": bounds
    }

    # Generate terrain & forest underlying continuous field (Perlin-like smooth noise)
    y, x = np.mgrid[0:height, 0:width]
    kx1, ky1 = rng.uniform(0.01, 0.03), rng.uniform(0.01, 0.03)
    kx2, ky2 = rng.uniform(0.03, 0.06), rng.uniform(0.03, 0.06)

    smooth_field = (
        0.6 * np.sin(kx1 * x + ky1 * y) +
        0.3 * np.cos(kx2 * x - ky2 * y) +
        0.1 * rng.standard_normal((height, width))
    )
    smooth_field = (smooth_field - smooth_field.min()) / (smooth_field.max() - smooth_field.min() + 1e-6)

    # 2021 Forest Probability & Binary Mask
    target_density = cfg["forest_density"]
    threshold_val = np.percentile(smooth_field, (1.0 - target_density) * 100)
    prob_2021 = 1.0 / (1.0 + np.exp(-12.0 * (smooth_field - threshold_val)))
    tree_2021 = (prob_2021 >= 0.5).astype(np.float32)

    # Deforestation loss zones (clustered patches in 2024)
    loss_centers = rng.integers(30, min(height, width) - 30, size=(6, 2))
    loss_mask = np.zeros((height, width), dtype=bool)
    for cr, cc in loss_centers:
        radius = rng.integers(12, 28)
        dist_sq = (y - cr)**2 + (x - cc)**2
        patch = (dist_sq <= radius**2) & (tree_2021 == 1)
        # 80% loss in patch
        patch &= (rng.random((height, width)) < 0.85)
        loss_mask |= patch

    # Regrowth gain zones
    gain_centers = rng.integers(30, min(height, width) - 30, size=(4, 2))
    gain_mask = np.zeros((height, width), dtype=bool)
    for cr, cc in gain_centers:
        radius = rng.integers(8, 18)
        dist_sq = (y - cr)**2 + (x - cc)**2
        patch = (dist_sq <= radius**2) & (tree_2021 == 0)
        gain_mask |= patch

    # 2024 Forest
    tree_2024 = tree_2021.copy()
    tree_2024[loss_mask] = 0.0
    tree_2024[gain_mask] = 1.0

    prob_2024 = prob_2021.copy()
    prob_2024[loss_mask] = np.clip(rng.uniform(0.05, 0.25, size=np.sum(loss_mask)), 0, 1)
    prob_2024[gain_mask] = np.clip(rng.uniform(0.75, 0.95, size=np.sum(gain_mask)), 0, 1)

    # Spectral bands (B2 Blue, B3 Green, B4 Red, B8 NIR)
    # Forest: Low Red (0.02-0.04), High NIR (0.25-0.38)
    # Non-Forest/Soil: Higher Red (0.08-0.16), Moderate NIR (0.12-0.20)
    red_2021 = np.where(tree_2021 == 1, rng.uniform(0.02, 0.045, (height, width)), rng.uniform(0.07, 0.15, (height, width)))
    nir_2021 = np.where(tree_2021 == 1, rng.uniform(0.24, 0.36, (height, width)), rng.uniform(0.10, 0.18, (height, width)))
    green_2021 = np.where(tree_2021 == 1, rng.uniform(0.04, 0.07, (height, width)), rng.uniform(0.05, 0.09, (height, width)))
    blue_2021 = np.where(tree_2021 == 1, rng.uniform(0.02, 0.04, (height, width)), rng.uniform(0.04, 0.07, (height, width)))

    red_2024 = np.where(tree_2024 == 1, rng.uniform(0.02, 0.045, (height, width)), rng.uniform(0.07, 0.15, (height, width)))
    nir_2024 = np.where(tree_2024 == 1, rng.uniform(0.24, 0.36, (height, width)), rng.uniform(0.10, 0.18, (height, width)))
    green_2024 = np.where(tree_2024 == 1, rng.uniform(0.04, 0.07, (height, width)), rng.uniform(0.05, 0.09, (height, width)))
    blue_2024 = np.where(tree_2024 == 1, rng.uniform(0.02, 0.04, (height, width)), rng.uniform(0.04, 0.07, (height, width)))

    s2_2021_rgb = np.stack([red_2021, green_2021, blue_2021], axis=-1).astype(np.float32)
    s2_2024_rgb = np.stack([red_2024, green_2024, blue_2024], axis=-1).astype(np.float32)

    ndvi_2021 = (nir_2021 - red_2021) / (nir_2021 + red_2021 + 1e-6)
    ndvi_2024 = (nir_2024 - red_2024) / (nir_2024 + red_2024 + 1e-6)
    ndvi_diff = ndvi_2024 - ndvi_2021

    # 4-class change
    change_map = np.zeros((height, width), dtype=np.float32)
    change_map[(tree_2021 == 1) & (tree_2024 == 1)] = 1  # Stable tree
    change_map[(tree_2021 == 1) & (tree_2024 == 0)] = 2  # Loss
    change_map[(tree_2021 == 0) & (tree_2024 == 1)] = 3  # Gain

    hotspot_map = (change_map == 2).astype(np.float32)

    return {
        "tree_cover_2021": tree_2021,
        "tree_cover_2024": tree_2024,
        "probability_2021": prob_2021.astype(np.float32),
        "probability_2024": prob_2024.astype(np.float32),
        "change_map": change_map,
        "ndvi_2021": ndvi_2021.astype(np.float32),
        "ndvi_2024": ndvi_2024.astype(np.float32),
        "ndvi_diff": ndvi_diff.astype(np.float32),
        "hotspot_map": hotspot_map,
        "rgb_2021": s2_2021_rgb,
        "rgb_2024": s2_2024_rgb,
        "metadata": meta,
        "config": cfg
    }


def save_sample_geotiffs_to_disk(output_dir="sample_tifs"):
    """Saves sample GeoTIFF files so users can also test uploading them."""
    os.makedirs(output_dir, exist_ok=True)
    created_files = []

    for district in ["Pune", "Satara", "Sindhudurg"]:
        data = generate_synthetic_district_data(district)
        meta = data["metadata"]
        transform = meta["transform"]

        def _save(arr, filename, dtype="float32", nodata=-9999):
            path = os.path.join(output_dir, filename)
            profile = {
                "driver": "GTiff",
                "height": meta["height"],
                "width": meta["width"],
                "count": 1,
                "dtype": dtype,
                "crs": meta["crs"],
                "transform": transform,
                "nodata": nodata,
                "compress": "lzw"
            }
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(arr.astype(dtype), 1)
            created_files.append(path)

        _save(data["tree_cover_2021"], f"{district}_2021_tree_cover.tif", "uint8", 255)
        _save(data["tree_cover_2024"], f"{district}_2024_tree_cover.tif", "uint8", 255)
        _save(data["probability_2021"], f"{district}_2021_probability.tif", "float32", -9999)
        _save(data["probability_2024"], f"{district}_2024_probability.tif", "float32", -9999)
        _save(data["change_map"], f"{district}_change_2021_2024.tif", "uint8", 255)
        _save(data["ndvi_diff"], f"{district}_NDVI_change_2021_2024.tif", "float32", -9999)

    return created_files

