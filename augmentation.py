"""
Data Augmentation Pipeline for Multi-Spectral Satellite Image Segmentation.
Dramatically expands small training sets (e.g. 32 patches -> 256+ patches)
and enforces spatial & illumination invariance.
"""

import numpy as np

def augment_patch_numpy(image, label, valid=None, rng=None):
    """
    Applies random geometric and spectral augmentations to a single patch:
    - Horizontal Flip (50% prob)
    - Vertical Flip (50% prob)
    - 90-degree Rotations (0, 90, 180, or 270 deg)
    - Subtle brightness & contrast scaling (within +/- 8%)
    """
    if rng is None:
        rng = np.random.default_rng()

    img_aug = image.copy()
    lbl_aug = label.copy()
    val_aug = valid.copy() if valid is not None else None

    # 1. Random horizontal flip
    if rng.random() > 0.5:
        img_aug = np.fliplr(img_aug)
        lbl_aug = np.fliplr(lbl_aug)
        if val_aug is not None:
            val_aug = np.fliplr(val_aug)

    # 2. Random vertical flip
    if rng.random() > 0.5:
        img_aug = np.flipud(img_aug)
        lbl_aug = np.flipud(lbl_aug)
        if val_aug is not None:
            val_aug = np.flipud(val_aug)

    # 3. Random 90 deg rotation
    k = rng.integers(0, 4)
    if k > 0:
        img_aug = np.rot90(img_aug, k, axes=(0, 1))
        lbl_aug = np.rot90(lbl_aug, k, axes=(0, 1))
        if val_aug is not None:
            val_aug = np.rot90(val_aug, k, axes=(0, 1))

    # 4. Subtle brightness/contrast jitter
    alpha = rng.uniform(0.92, 1.08)  # contrast
    beta = rng.uniform(-0.04, 0.04)  # brightness
    img_aug = np.clip(img_aug * alpha + beta, 0.0, 1.0)

    if val_aug is not None:
        return img_aug, lbl_aug, val_aug
    return img_aug, lbl_aug


def expand_dataset_with_augmentations(X, Y, V, target_count=256, seed=42):
    """
    Takes initial training patches and generates an expanded augmented dataset
    including deterministic geometric invariants (D4 group: 8 symmetries)
    plus randomized perturbations.
    """
    rng = np.random.default_rng(seed)
    n_original = len(X)
    if n_original == 0:
        return X, Y, V

    augmented_X = [X]
    augmented_Y = [Y]
    augmented_V = [V]

    # Deterministic D4 symmetry expansion (8x expansion)
    # 1: rot90
    augmented_X.append(np.rot90(X, 1, axes=(1, 2)))
    augmented_Y.append(np.rot90(Y, 1, axes=(1, 2)))
    augmented_V.append(np.rot90(V, 1, axes=(1, 2)))

    # 2: rot180
    augmented_X.append(np.rot90(X, 2, axes=(1, 2)))
    augmented_Y.append(np.rot90(Y, 2, axes=(1, 2)))
    augmented_V.append(np.rot90(V, 2, axes=(1, 2)))

    # 3: rot270
    augmented_X.append(np.rot90(X, 3, axes=(1, 2)))
    augmented_Y.append(np.rot90(Y, 3, axes=(1, 2)))
    augmented_V.append(np.rot90(V, 3, axes=(1, 2)))

    # 4: horizontal flip
    X_flip_h = np.flip(X, axis=2)
    Y_flip_h = np.flip(Y, axis=2)
    V_flip_h = np.flip(V, axis=2)
    augmented_X.append(X_flip_h)
    augmented_Y.append(Y_flip_h)
    augmented_V.append(V_flip_h)

    # 5: vertical flip
    X_flip_v = np.flip(X, axis=1)
    Y_flip_v = np.flip(Y, axis=1)
    V_flip_v = np.flip(V, axis=1)
    augmented_X.append(X_flip_v)
    augmented_Y.append(Y_flip_v)
    augmented_V.append(V_flip_v)

    # 6: transpose / diagonal flip
    augmented_X.append(np.swapaxes(X, 1, 2))
    augmented_Y.append(np.swapaxes(Y, 1, 2))
    augmented_V.append(np.swapaxes(V, 1, 2))

    # 7: anti-diagonal flip
    augmented_X.append(np.swapaxes(np.rot90(X, 2, axes=(1, 2)), 1, 2))
    augmented_Y.append(np.swapaxes(np.rot90(Y, 2, axes=(1, 2)), 1, 2))
    augmented_V.append(np.swapaxes(np.rot90(V, 2, axes=(1, 2)), 1, 2))

    # Concatenate deterministic symmetries (8 * n_original)
    X_d4 = np.concatenate(augmented_X, axis=0)
    Y_d4 = np.concatenate(augmented_Y, axis=0)
    V_d4 = np.concatenate(augmented_V, axis=0)

    # If still below target_count, synthesize randomized jittered patches
    if len(X_d4) < target_count:
        needed = target_count - len(X_d4)
        extra_X, extra_Y, extra_V = [], [], []
        for _ in range(needed):
            idx = rng.integers(0, n_original)
            img, lbl, val = augment_patch_numpy(X[idx], Y[idx], V[idx], rng=rng)
            extra_X.append(img)
            extra_Y.append(lbl)
            extra_V.append(val)
        X_all = np.concatenate([X_d4, np.array(extra_X)], axis=0)
        Y_all = np.concatenate([Y_d4, np.array(extra_Y)], axis=0)
        V_all = np.concatenate([V_d4, np.array(extra_V)], axis=0)
    else:
        X_all, Y_all, V_all = X_d4, Y_d4, V_d4

    # Shuffle synchronously
    perm = rng.permutation(len(X_all))
    return X_all[perm], Y_all[perm], V_all[perm]

