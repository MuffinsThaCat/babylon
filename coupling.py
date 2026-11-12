"""Babylon: coupling analysis for climate tipping point cascades.

TC, bill, hierarchical decomposition, halving depth D, capacity line at 3.03.
Applied to climate tipping element interactions.

Requires only numpy.
"""
import numpy as np


# ============================================================================
# Total correlation
# ============================================================================

def tc_gaussian(S):
    """TC = sum_i 1/2 ln(S_ii) - 1/2 ln det S  (nats)."""
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return np.inf
    return 0.5 * (np.log(np.diag(S)).sum() - logdet)


def per_component_bill(S):
    """DeltaTC_i = 1/2 (ln Sigma_ii + ln Omega_ii), Omega = Sigma^{-1}.
    Ranks elements by coupling contribution."""
    Omega = np.linalg.inv(S)
    return 0.5 * (np.log(np.maximum(np.diag(S), 1e-300))
                  + np.log(np.maximum(np.diag(Omega), 1e-300)))


# ============================================================================
# Correlation → covariance for coupling analysis
# ============================================================================

def cov_from_coupling(C, q=0.9):
    """Sigma = (I - c*C)^{-1} with c = q / ||C||_2.  PD for q < 1."""
    n = C.shape[0]
    c = q / max(np.linalg.norm(C, 2), 1e-15)
    return np.linalg.inv(np.eye(n) - c * C)


def correlation_to_coupling(corr, threshold=0.6):
    """Convert return correlation matrix to coupling matrix.

    Keeps only correlations above threshold (analogous to STRING >= 0.9
    in drug discovery — filters the noise floor of sector-wide co-movement).
    Negative correlations are hedges, not risk. Zeroes the diagonal.

    threshold=0.6 is the financial equivalent of STRING high-confidence:
    only pairs with economically meaningful coupling survive.
    """
    C = np.maximum(corr, 0.0)
    C[C < threshold] = 0.0
    np.fill_diagonal(C, 0.0)
    return C


def regularize(S, eps=1e-6):
    """Shrink S toward identity until min eigenvalue >= eps."""
    lam_min = np.linalg.eigvalsh(S)[0]
    if lam_min >= eps:
        return S, 0.0
    t = (eps - lam_min) / (1.0 - lam_min + 1e-15)
    t = max(t, eps)
    return (1 - t) * S + t * np.eye(S.shape[0]), t


# ============================================================================
# Hierarchical decomposition
# ============================================================================

def _phase1(C):
    """Phase 1: merge by mean coupling, module size <= 3."""
    n = C.shape[0]
    S = np.array(C, dtype=float)
    sz = [1] * n
    mods = [[i] for i in range(n)]
    while True:
        k = len(mods)
        s = np.array(sz, dtype=float)
        M = S / np.outer(s, s)
        ok = ((s[:, None] + s[None, :] <= 3)
              & np.triu(np.ones((k, k), dtype=bool), 1)
              & (M > 0))
        if not ok.any():
            break
        vals = np.where(ok, M, -1.0)
        mx = vals.max()
        x, y = divmod(int(np.argmax(vals >= mx - 1e-12)), k)
        mods[x] = mods[x] + mods[y]
        del mods[y]
        S[x, :] += S[y, :]
        S[:, x] += S[:, y]
        S = np.delete(np.delete(S, y, 0), y, 1)
        sz[x] += sz[y]
        del sz[y]
    return mods


def compile_gap(C):
    """Hierarchical decomposition: Phase 1 then Phase 2 greedy pairing.
    Returns list of (fraction, partition) at each level."""
    C = np.array(C, dtype=float)
    n = C.shape[0]
    Cl = C.tolist()
    mods = _phase1(C)
    total = float(np.sum(np.triu(C, k=1)))

    def gsum(a, b):
        return sum(Cl[i][j] for i in a for j in b)

    def inter(groups):
        return sum(gsum(groups[a], groups[b])
                   for a in range(len(groups))
                   for b in range(a + 1, len(groups)))

    prev = inter(mods)
    res = [(prev / total if total > 0 else 0.0, [list(m) for m in mods])]
    groups = [list(m) for m in mods]

    while len(groups) > 2 and prev > 0:
        pairs = sorted(
            ((gsum(groups[a], groups[b]), a, b)
             for a in range(len(groups))
             for b in range(a + 1, len(groups))),
            reverse=True,
        )
        used, new = set(), []
        for _, a, b in pairs:
            if a in used or b in used:
                continue
            used |= {a, b}
            new.append(groups[a] + groups[b])
        new += [groups[i] for i in range(len(groups)) if i not in used]
        cur = inter(new)
        res.append((cur / prev, new))
        prev, groups = cur, new

    return res


# ============================================================================
# Residuals and halving depth
# ============================================================================

def node_residuals(C, parts):
    """Per-node and global residual at each partition level."""
    C2 = C ** 2
    s = C2.sum(1)
    s = np.where(s > 0, s, 1.0)
    n = C.shape[0]
    K = len(parts)
    Rn = np.empty((K, n))
    R = np.empty(K)
    for k, g in enumerate(parts):
        lab = np.empty(n, int)
        for j, grp in enumerate(g):
            for idx in grp:
                lab[idx] = j
        cross = C2 * (lab[:, None] != lab[None, :])
        Rn[k] = cross.sum(1) / s
        denom = C2.sum()
        R[k] = cross.sum() / denom if denom > 0 else 0.0
    return Rn, R


def halving_depth(R):
    """Levels to bring R(k) to 1/4.
    D < 3.03 → modular (failure containable).
    D > 3.03 → globally entangled (cascading failure risk)."""
    prev = 1.0
    for k in range(len(R)):
        cur = R[k]
        if cur <= 0.25:
            if cur == prev:
                return float(k), True
            t = ((np.log(0.25) - np.log(prev))
                 / (np.log(max(cur, 1e-300)) - np.log(prev)))
            return k + t, (k < len(R) - 1)
        prev = cur
    return np.inf, False
