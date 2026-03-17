"""
Fast, robust 2D star-list matching (SciPy-only, single-core).

Given two 2D point sets (reference, input), estimate a similarity transform
(scale + rotation + translation) that maps reference coordinates into the
input frame and return the matched index pairs.

Pipeline overview
- Delaunay triangles: build on the first ``tri_build_points`` points of each set.
- Triangle space: embed each triangle as two normalized side ratios and keep
  its side-order and orientation parity (CCW/CW).
- Triangle matching: mutual 1-NN in triangle space (KDTree) within ``tri_match_radius``
  and sort by distance to prefer tighter correspondences.
- Voting: each matched triangle pair votes for its three vertex correspondences
  (respecting side-order). Greedily pick a conflict-free set of high-vote pairs.
- Seed + prune: fit Umeyama on the top ``seed_size`` pairs, prune by a robust
  sigma rule (first pass without a tight cap), re-fit.
- Final refine: mutual 1-NN in point space followed by a final Umeyama fit.

Design notes
- SciPy-only (no numba), single-core KDTree queries for predictable behavior.
- Orientation parity is enforced (CCW only); mirrored solutions are not matched
  unless you modify the parity filters or allow reflection in the fit.
- The initial fit uses a compact seed (``seed_size``) to resist heavy outliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay, cKDTree


@dataclass(frozen=True)
class Similarity:
    """2D similarity transform.

    The transform maps a point ``x`` in the reference frame to the input frame
    as ``x' = A @ x + t`` where ``A = s * R`` is a scaled rotation (uniform
    scale, no shear) and ``t`` is a translation.
    """

    A: NDArray  # shape (2,2)
    t: NDArray  # shape (2,)

    @property
    def scale(self) -> float:
        c0 = float(np.linalg.norm(self.A[:, 0]))
        c1 = float(np.linalg.norm(self.A[:, 1]))
        return 0.5 * (c0 + c1)

    @property
    def rotation(self) -> float:
        s = max(np.linalg.norm(self.A[:, 0]), np.linalg.norm(self.A[:, 1]))
        if s <= 0:
            return 0.0
        R = self.A / s
        return np.atan2(float(R[1, 0]), float(R[0, 0]))

    def to_matrix3x3(self) -> NDArray:
        M = np.eye(3, dtype=float)
        M[:2, :2] = self.A
        M[:2, 2] = self.t
        return M

    def apply(self, xy: NDArray) -> NDArray:
        return (xy @ self.A.T) + self.t


@dataclass
class MatchResult:
    """Result of star-list matching.

    - ``transform``: estimated similarity transform mapping ref → input.
    - ``pairs``: array of index pairs into (ref, input), shape (K, 2).
    - ``rms``: RMS error in input frame across ``pairs`` after the final fit.
    - ``inlier_count``: number of matched pairs (``pairs.shape[0]``).
    """

    transform: Similarity
    pairs: NDArray  # shape (K, 2) indices into (ref, inp)
    rms: float
    inlier_count: int


def _triangle_space(x: NDArray, y: NDArray, tris: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute triangle-space coordinates and metadata.

    For each triangle (i, j, k):
    - Compute squared side lengths ``s0, s1, s2`` opposite vertices 0, 1, 2.
    - Sort sides to obtain the order indices ``ords`` (ascending length).
    - Map to triangle space as ``(b/a, c/a) = (sqrt(s_mid/s_max), sqrt(s_min/s_max))``.
    - Compute orientation parity ``ori`` as +1 (CCW) or -1 (CW).
    Returns ``(b_over_a, c_over_a, ords, ori)``.
    """

    i = tris[:, 0]; j = tris[:, 1]; k = tris[:, 2]
    s0 = (x[j] - x[k]) ** 2 + (y[j] - y[k]) ** 2
    s1 = (x[k] - x[i]) ** 2 + (y[k] - y[i]) ** 2
    s2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
    S = np.stack([s0, s1, s2], axis=1)
    ords = np.argsort(S, axis=1)
    smin = S[np.arange(len(S)), ords[:, 0]]
    smid = S[np.arange(len(S)), ords[:, 1]]
    smax = S[np.arange(len(S)), ords[:, 2]]
    with np.errstate(divide="ignore", invalid="ignore"):
        b_over_a = np.sqrt(np.maximum(smid, 0) / np.maximum(smax, 1e-32))
        c_over_a = np.sqrt(np.maximum(smin, 0) / np.maximum(smax, 1e-32))
    v1x = x[j] - x[i]; v1y = y[j] - y[i]
    v2x = x[k] - x[i]; v2y = y[k] - y[i]
    cross = v1x * v2y - v1y * v2x
    ori = np.where(cross >= 0.0, 1, -1)
    return b_over_a, c_over_a, ords.astype(np.int64), ori.astype(np.int64)


def _kdtree_mutual_nn(
    A: NDArray,
    B: NDArray,
    radius: Optional[float] = None,
    *,
    sort_by_distance: bool = False,
) -> NDArray:
    """Mutual nearest neighbors between ``A`` and ``B`` using KDTree.

    - ``radius``: if provided, keep only forward matches with distance <= radius.
    - ``sort_by_distance``: if True, order returned pairs by increasing forward
      distance. This is useful when subsequent steps weight earlier pairs more.
    Returns an array of shape (K, 2) with indices into (A, B).
    """
    if len(A) == 0 or len(B) == 0:
        return np.empty((0, 2), dtype=np.int64)
    treeB = cKDTree(B)
    d_f, j_f = treeB.query(A, k=1)
    treeA = cKDTree(A)
    _, i_b = treeA.query(B, k=1)
    mask = (i_b[j_f] == np.arange(A.shape[0]))
    if radius is not None:
        mask &= (d_f <= radius)
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int64)
    i_idx = np.nonzero(mask)[0]
    j_idx = j_f[mask].astype(np.int64)
    if sort_by_distance:
        order = np.argsort(d_f[mask])
        i_idx = i_idx[order]
        j_idx = j_idx[order]
    return np.stack([i_idx, j_idx], axis=1).astype(np.int64)


def _accumulate_votes(
    ref_tris: NDArray,
    ref_ord: NDArray,
    inp_tris: NDArray,
    inp_ord: NDArray,
    matches: NDArray,
    n_ref: int,
    n_inp: int,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Accumulate votes from matched triangle pairs into a compact grid.

    For each matched triangle pair, cast three votes for the corresponding
    vertices indexed by the side-order mapping. Votes are weighted so that
    earlier triangle matches (e.g., closer in triangle space) contribute more.
    Returns ``(votes, used_ref, used_inp)``, where ``votes`` is a dense grid
    for the compacted sets ``used_ref`` and ``used_inp``.
    """
    if matches.size == 0:
        return np.zeros((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    used_ref = np.unique(ref_tris[matches[:, 0]].ravel())
    used_inp = np.unique(inp_tris[matches[:, 1]].ravel())
    map_r = -np.ones(n_ref, dtype=np.int64)
    map_i = -np.ones(n_inp, dtype=np.int64)
    map_r[used_ref] = np.arange(len(used_ref), dtype=np.int64)
    map_i[used_inp] = np.arange(len(used_inp), dtype=np.int64)
    votes = np.zeros((len(used_ref), len(used_inp)), dtype=np.int64)
    m = matches.shape[0]
    for rank, (tr, ti) in enumerate(matches):
        w = m - rank
        r0, r1, r2 = ref_tris[tr]
        i0, i1, i2 = inp_tris[ti]
        for pos in range(3):
            sref = ref_ord[tr, pos]
            sinp = inp_ord[ti, pos]
            vref = (r0 if sref == 0 else (r1 if sref == 1 else r2))
            vinp = (i0 if sinp == 0 else (i1 if sinp == 1 else i2))
            lr = map_r[vref]; li = map_i[vinp]
            if lr >= 0 and li >= 0:
                votes[lr, li] += w
    return votes, used_ref, used_inp


def _prune_limit(d2_sorted: NDArray, md2_soft: float) -> int:
    """Compute prune limit with a robust sigma rule.

    Uses the 40th percentile of the sorted squared residuals as a robust scale
    estimate, with an upper bound of ``md2_soft`` (when finite). Falls back to
    a small minimum when too strict.
    """
    n = d2_sorted.shape[0]
    if n == 0:
        return 0
    k40 = min(n - 1, int(0.4 * n + 0.5))
    sigma = d2_sorted[k40]
    if sigma < 1e-6:
        return n
    thresh = min(36.0 * sigma, md2_soft)
    lim = int(np.searchsorted(d2_sorted, thresh, side="right"))
    if lim <= 0:
        lim = min(10, n)
    return lim


def _umeyama_similarity(src: NDArray, dst: NDArray, allow_reflection: bool = False) -> Similarity:
    """Umeyama's similarity fit (uniform scale, no shear).

    Parameters
    - ``src``, ``dst``: shape (N, 2). If ``allow_reflection`` is False and the
      SVD suggests a reflection, the reflection component is flipped to keep a
      proper rotation.
    """
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError("src and dst must be (N,2) arrays with same shape")
    n = src.shape[0]
    if n == 0:
        raise ValueError("no points to fit")
    mu_x = np.mean(src, axis=0)
    mu_y = np.mean(dst, axis=0)
    x = src - mu_x
    y = dst - mu_y
    Sigma = (y.T @ x) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    det = np.linalg.det(U @ Vt)
    if det < 0 and not allow_reflection:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_x = np.mean(np.sum(x * x, axis=1))
    c = (np.trace(np.diag(D) @ S)) / (var_x + 1e-32)
    A = c * R
    t = mu_y - (A @ mu_x)
    return Similarity(A=A, t=t)


def match_stars(
    inp_xy: NDArray,
    ref_xy: NDArray,
    max_distance: float,
    *,
    tri_build_points: int = 2048,
    tri_match_radius: float = 0.03,
    seed_size: int = 80,
) -> MatchResult:
    """Match two 2D point sets and estimate a similarity transform.

    Parameters
    - ``inp_xy``: (N2, 2) input coordinates.
    - ``ref_xy``: (N1, 2) reference coordinates.
    - ``max_distance``: acceptance radius (in input frame) for final symmetric
      mutual-NN matching.
    - ``tri_build_points``: number of points to include when building Delaunay
      triangles (larger increases coverage and cost). Default 1024.
    - ``tri_match_radius``: tolerance in triangle space for mutual matching;
      smaller is stricter, larger is more permissive. Default 0.03.
    - ``seed_size``: number of top voted pairs to seed the initial fit. Smaller
      values improve robustness under heavy outliers; larger values may help
      when outliers are rare. Default 80.


    Notes
    - For very small N, tolerance is automatically relaxed in triangle space.
    - Only CCW triangle parity is used on both sides (no reflection).
    - If Delaunay triangulation fails to produce triangles in either set, the
      function falls back to a translation-only guess and mutual-NN.
    """
    if inp_xy.ndim != 2 or ref_xy.ndim != 2 or inp_xy.shape[1] != 2 or ref_xy.shape[1] != 2:
        raise ValueError("ref_xy and inp_xy must be (N,2) arrays")
    if len(inp_xy) < 3 or len(ref_xy) < 3:
        A = np.eye(2)
        t = np.mean(ref_xy, axis=0) - np.mean(inp_xy, axis=0)
        T = Similarity(A=A, t=t)
        pairs = _kdtree_mutual_nn(T.apply(inp_xy), ref_xy, radius=max_distance)
        rms = float(np.sqrt(np.mean(np.sum((T.apply(inp_xy[pairs[:, 0]]) - ref_xy[pairs[:, 1]]) ** 2, axis=1)))) if len(pairs) else float("inf")
        return MatchResult(transform=T, pairs=pairs, rms=rms, inlier_count=len(pairs))
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")

    # Delaunay triangles on a subset of points
    n_ref_tri = min(len(inp_xy), tri_build_points)
    n_inp_tri = min(len(ref_xy), tri_build_points)
    ref_tris = Delaunay(inp_xy[:n_ref_tri]).simplices.astype(np.int64) if n_ref_tri >= 3 else np.empty((0, 3), dtype=np.int64)
    inp_tris = Delaunay(ref_xy[:n_inp_tri]).simplices.astype(np.int64) if n_inp_tri >= 3 else np.empty((0, 3), dtype=np.int64)
    if ref_tris.size == 0 or inp_tris.size == 0:
        A = np.eye(2); t = np.mean(ref_xy, axis=0) - np.mean(inp_xy, axis=0)
        T = Similarity(A=A, t=t)
        pairs = _kdtree_mutual_nn(T.apply(inp_xy), ref_xy, radius=max_distance)
        rms = float(np.sqrt(np.mean(np.sum((T.apply(inp_xy[pairs[:, 0]]) - ref_xy[pairs[:, 1]]) ** 2, axis=1)))) if len(pairs) else float("inf")
        return MatchResult(transform=T, pairs=pairs, rms=rms, inlier_count=len(pairs))

    # Triangle-space mapping
    ref_ba, ref_ca, ref_ord, ref_parity = _triangle_space(inp_xy[:, 0], inp_xy[:, 1], ref_tris)
    inp_ba, inp_ca, inp_ord, inp_parity = _triangle_space(ref_xy[:, 0], ref_xy[:, 1], inp_tris)

    best_T = None
    best_pairs = None
    # Loosen triangle-space tolerance for small-N (heuristic scaling)
    tri_radius_eff = float(tri_match_radius)
    if n_ref_tri < 300:
        tri_radius_eff *= 3.0
    if n_ref_tri < 200:
        tri_radius_eff *= 4.0
    if n_ref_tri < 100:
        tri_radius_eff *= 6.0

    # Symmetric matching in triangle-space (mutual 1-NN)
    ref_ccw = (ref_parity == 1)
    inp_ccw = (inp_parity == 1)
    ref_ts = np.column_stack([ref_ba[ref_ccw], ref_ca[ref_ccw]])
    inp_ts = np.column_stack([inp_ba[inp_ccw], inp_ca[inp_ccw]])
    ref_tris_ccw = ref_tris[ref_ccw]
    inp_tris_ccw = inp_tris[inp_ccw]
    ref_ord_ccw = ref_ord[ref_ccw]
    inp_ord_ccw = inp_ord[inp_ccw]
    tri_matches = _kdtree_mutual_nn(ref_ts, inp_ts, radius=tri_radius_eff, sort_by_distance=True)
    # Vote accumulation and greedy conflict-free selection
    vote_grid, used_ref, used_inp = _accumulate_votes(ref_tris_ccw, ref_ord_ccw, inp_tris_ccw, inp_ord_ccw, tri_matches, len(inp_xy), len(ref_xy))
    nz = np.argwhere(vote_grid > 0)
    vals = vote_grid[vote_grid > 0]
    order = np.argsort(-vals)
    used_ref_mask = np.zeros(len(inp_xy), dtype=bool)
    used_inp_mask = np.zeros(len(ref_xy), dtype=bool)
    vote_pairs = []
    for pos in order:
        lr = int(nz[pos, 0]); li = int(nz[pos, 1])
        r = int(used_ref[lr]); j = int(used_inp[li])
        if used_ref_mask[r] or used_inp_mask[j]:
            continue
        vote_pairs.append((r, j))
        used_ref_mask[r] = True
        used_inp_mask[j] = True
    vote_pairs = np.array(vote_pairs, dtype=int)
    # Use a compact, high-confidence seed to estimate the initial transform.
    seed_size_eff = min(int(seed_size), vote_pairs.shape[0])
    seed_pairs = vote_pairs[: seed_size_eff]
    for it in range(2):
        src = inp_xy[seed_pairs[:, 0]]
        dst = ref_xy[seed_pairs[:, 1]]
        Ttmp = _umeyama_similarity(src, dst)
        d2 = np.sum((Ttmp.apply(src) - dst) ** 2, axis=1)
        order_d = np.argsort(d2)
        # Prune: first pass without tight cap, second pass with soft cap
        md2_soft = np.inf if it == 0 else (max_distance * 2.0) ** 2
        lim = _prune_limit(d2[order_d], md2_soft)
        # Ensure we keep enough pairs to avoid degenerate refits
        lim = max(lim, min(20, seed_pairs.shape[0]))
        seed_pairs = seed_pairs[order_d[:lim]]
        if seed_pairs.shape[0] < 3:
            break
    T0 = _umeyama_similarity(inp_xy[seed_pairs[:, 0]], ref_xy[seed_pairs[:, 1]])
    pairs = _kdtree_mutual_nn(T0.apply(inp_xy), ref_xy, radius=max_distance)
    if pairs.shape[0] >= 3:
        T1 = _umeyama_similarity(inp_xy[pairs[:, 0]], ref_xy[pairs[:, 1]])
        pairs = _kdtree_mutual_nn(T1.apply(inp_xy), ref_xy, radius=max_distance)
        best_T = T1
        best_pairs = pairs
    else:
        best_T = T0
        best_pairs = pairs

    if best_pairs.size > 0:
        err = best_T.apply(inp_xy[best_pairs[:, 0]]) - ref_xy[best_pairs[:, 1]]
        rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    else:
        rms = float("inf")
    return MatchResult(transform=best_T, pairs=best_pairs, rms=rms, inlier_count=best_pairs.shape[0])


__all__ = ["Similarity", "MatchResult", "match_stars"]
