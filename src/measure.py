"""Objective 2 — automatic plaque / stenosis measurement (QCA-style) inside a detected box.

Background
----------
Invasive coronary angiography is a 2-D X-ray projection of the contrast-filled LUMEN.
The plaque itself is not visible; only the narrowing it causes is. Following
Quantitative Coronary Angiography (QCA), plaque burden is measured as the
difference between the reconstructed healthy (reference) vessel and the actual lumen.

Algorithm
---------
 1. crop around the detection, upsample x2, black top-hat → "vesselness" map
 2. CENTRELINE: minimum-cost path through the dark vessel that crosses the
    detection box (bridges the faint lumen at tight stenoses)
 3. PROFILES: intensity profiles perpendicular to the smoothed centreline
 4. EDGE DIAMETER  D(s): distance between the max-gradient edges of each profile
 5. DENSITOMETRY   A(s): integral of -log(I / I_background) across the profile
    (Beer-Lambert ⇒ proportional to cross-sectional lumen area; independent of
    lumen shape and still valid where edges are blurred at tight stenoses)
 6. REFERENCE: robust straight line through the healthy segments proximal and distal
    to the box (interpolated reference, as in QCA) for both D and A
 7. OUTPUTS
      MLD                minimal lumen diameter (edge)
      %DS                percent diameter stenosis — default 'hybrid' = mean of the
                         edge and densitometric diameter ratios (robust to vessel
                         overlap); both components are also reported
      %AS                percent area stenosis (densitometric)
      lesion length      arc length where lumen < 0.8 x reference around the MLD
      plaque area (2-D)  integral of (Dref - D) ds   — projected plaque area
      plaque volume      integral of pi/4 Dref^2 (1 - A/Aref) ds — ESTIMATE from a
                         single projection (true volume needs IVUS/OCT/CCTA)
 Values are in pixels; with mm_per_pixel (catheter calibration or DICOM
 PixelSpacing) they are also given in mm, mm^2, mm^3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import medfilt
from skimage import feature, filters, graph, morphology


@dataclass
class Measurement:
    ok: bool
    reason: str = ""
    mld_px: float = np.nan
    ref_diameter_px: float = np.nan
    percent_diameter_stenosis: float = np.nan     # primary %DS (hybrid by default)
    percent_ds_edge: float = np.nan               # edge-based %DS
    percent_ds_densitometric: float = np.nan      # densitometric %DS = 1 - sqrt(A/Aref)
    percent_area_stenosis: float = np.nan         # densitometric %AS
    lesion_length_px: float = np.nan
    plaque_area_px2: float = np.nan
    plaque_volume_px3: float = np.nan
    mld_mm: float = np.nan
    ref_diameter_mm: float = np.nan
    lesion_length_mm: float = np.nan
    plaque_area_mm2: float = np.nan
    plaque_volume_mm3: float = np.nan
    severity: str = ""
    quality: float = np.nan                       # 0-1 heuristic confidence of the measurement
    centerline: list = field(default_factory=list, repr=False)
    edges: tuple = field(default=((), ()), repr=False)
    lesion_span: tuple = field(default=(0, 0), repr=False)
    mld_point: tuple = field(default=(0, 0), repr=False)
    profile: dict = field(default_factory=dict, repr=False)

    def row(self) -> dict:
        d = asdict(self)
        for k in ("centerline", "edges", "lesion_span", "mld_point", "profile"):
            d.pop(k)
        return d


def severity_label(ds: float) -> str:
    if not np.isfinite(ds):
        return "n/a"
    if ds < 50:
        return "mild (<50%)"
    if ds < 70:
        return "moderate (50-69%)"
    if ds < 90:
        return "severe (70-89%)"
    return "critical (>=90%)"


# ---------------------------------------------------------------- helpers
def _crop(gray, box, margin, min_crop, max_crop=224):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side = min(max(max(w, h) * (1 + 2 * margin), min_crop), max_crop)   # cap keeps latency bounded
    x0 = int(max(0, round(cx - side / 2)))
    y0 = int(max(0, round(cy - side / 2)))
    x1 = int(min(gray.shape[1], round(cx + side / 2)))
    y1 = int(min(gray.shape[0], round(cy + side / 2)))
    return gray[y0:y1, x0:x1], x0, y0


def _in_box(pts_yx, b, pad=0.0):
    x, y, w, h = b
    return ((pts_yx[:, 1] >= x - pad) & (pts_yx[:, 1] <= x + w + pad) &
            (pts_yx[:, 0] >= y - pad) & (pts_yx[:, 0] <= y + h + pad))


def _remove_small(mask, n):
    """remove_small_objects across scikit-image versions (keyword renamed min_size -> max_size)."""
    try:
        return morphology.remove_small_objects(mask, max_size=n)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=n)


def _robust_reference(arc, val, healthy, valid):
    """Interpolated reference: line through the upper 75 % of healthy values."""
    if healthy.sum() >= 6:
        hv = val[healthy]
        keep = hv >= np.percentile(hv, 25)
        if np.ptp(arc[healthy][keep]) > 0:
            coef = np.polyfit(arc[healthy][keep], hv[keep], 1)
            ref = np.polyval(coef, arc)
            return np.maximum(ref, np.median(hv) * 0.5)
    return np.full_like(val, np.percentile(val[valid], 85))


def _subpix(arr, i):
    if 0 < i < len(arr) - 1:
        a, b, c = arr[i - 1], arr[i], arr[i + 1]
        den = a - 2 * b + c
        if den != 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def _centerline(th, bl, s):
    """Minimum-cost path through the vessel crossing the box. Returns (N,2) yx or None."""
    thn = np.clip(th / (np.percentile(th, 99.5) + 1e-6), 0, 1)
    hi = filters.threshold_otsu(th) if th.max() > 0 else 0
    if hi <= 0:
        return None
    mask = filters.apply_hysteresis_threshold(th, 0.5 * hi, hi)
    mask = _remove_small(mask, int(30 * s * s))
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    x, y, w, h = bl
    pad = 0.5 * max(w, h)
    win = np.zeros_like(mask)
    win[int(max(0, y - pad)):int(y + h + pad) + 1, int(max(0, x - pad)):int(x + w + pad) + 1] = True
    keep = np.unique(lab[win & mask])
    keep = keep[keep > 0]
    if len(keep) == 0:
        return None
    sel = np.isin(lab, keep)
    skel = morphology.skeletonize(sel)
    pts = np.argwhere(skel)
    if len(pts) < 5:
        return None
    # anchor: strongest vessel response inside the box
    inbox = np.zeros_like(mask)
    inbox[int(max(0, y)):int(y + h) + 1, int(max(0, x)):int(x + w) + 1] = True
    sm = ndi.gaussian_filter(thn, s)
    cand = np.argwhere(inbox & sel) if (inbox & sel).any() else np.argwhere(inbox)
    anchor = np.array(cand[np.argmax(sm[tuple(cand.T)])])
    # local vessel direction at the anchor: Hessian eigenvector with the smallest |eigenvalue|
    sig = max(2.0, 0.25 * max(w, h))
    Hrr, Hrc, Hcc = feature.hessian_matrix(thn, sigma=sig, order="rc", use_gaussian_derivatives=False)
    ay, ax = anchor
    Hm = np.array([[Hrr[ay, ax], Hrc[ay, ax]], [Hrc[ay, ax], Hcc[ay, ax]]])
    ev, evec = np.linalg.eigh(Hm)
    v = evec[:, np.argmin(np.abs(ev))]                       # (dy, dx) along the vessel
    rel = pts - anchor
    dist = np.hypot(*rel.T)
    proj = rel @ v
    cos = proj / (dist + 1e-6)
    # end points: far along +v and -v, staying roughly aligned with the vessel axis
    sc = np.where(np.abs(cos) > 0.5, proj, 0)
    e1, e2 = tuple(pts[np.argmax(sc)]), tuple(pts[np.argmin(sc)])
    if np.hypot(e1[0] - e2[0], e1[1] - e2[1]) < 8 * s:
        e1 = tuple(pts[np.argmax(dist)])
        e2 = tuple(pts[np.argmax(np.hypot(*(pts - e1).T))])
        if np.hypot(e1[0] - e2[0], e1[1] - e2[1]) < 8 * s:
            return None
    cost = 1.0 / (thn + 0.02)
    a = tuple(anchor)
    p1, _ = graph.route_through_array(cost, e1, a, fully_connected=True, geometric=True)
    p2, _ = graph.route_through_array(cost, a, e2, fully_connected=True, geometric=True)
    path = np.array(p1 + p2[1:])
    return path, sel, len(p1) - 1


# ---------------------------------------------------------------- main API
def measure_stenosis(image: np.ndarray, box, mm_per_pixel: float | None = None,
                     margin: float = 1.0, min_crop: int = 96, lesion_threshold: float = 0.8,
                     upsample: int = 2, primary: str = "hybrid") -> Measurement:
    """Measure a stenosis inside `box` = (x, y, w, h) on a grayscale/BGR angiogram.

    primary: 'hybrid' (mean of edge & densitometric ratios, default), 'edge', 'densitometric'.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    crop, x0, y0 = _crop(gray, box, margin, min_crop)
    if crop.size == 0 or min(crop.shape) < 16:
        return Measurement(False, "empty crop")
    s = upsample
    cu = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    bl = ((box[0] - x0) * s, (box[1] - y0) * s, box[2] * s, box[3] * s)

    g = ndi.gaussian_filter(cu, 0.5 * s)
    # black top-hat (= closing - image) at native resolution, then upsampled (≈8x faster)
    rad = int(np.clip(0.9 * max(box[2], box[3]), 10, 28))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    g0 = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 0.5)
    th = cv2.resize(cv2.morphologyEx(g0, cv2.MORPH_BLACKHAT, se), (cu.shape[1], cu.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    res = _centerline(th, bl, s)
    if res is None:
        return Measurement(False, "vessel/centreline not found")
    path, sel, ia = res
    edt = ndi.distance_transform_edt(sel)

    # resample + smooth centreline (1 upsampled px spacing)
    p = path.astype(float)
    a = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    # keep only the segment near the lesion (QCA uses nearby proximal/distal references)
    d_est = 2 * np.median(edt[tuple(path.T)]) + 1
    lmax = max(2.5 * max(bl[2], bl[3]), 6 * d_est)
    keep = np.abs(a - a[ia]) <= lmax
    p, a = p[keep], a[keep] - a[keep][0]
    if a[-1] < 10 * s:
        return Measurement(False, "centreline too short")
    arc = np.arange(0, a[-1], 1.0)
    p = np.c_[np.interp(arc, a, p[:, 0]), np.interp(arc, a, p[:, 1])]
    p = ndi.gaussian_filter1d(p, 2.0 * s, axis=0, mode="nearest")
    t = np.gradient(p, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.c_[t[:, 1], -t[:, 0]]                          # (ny, nx)

    d0 = 2 * np.median(edt[np.clip(p[:, 0].round().astype(int), 0, sel.shape[0] - 1),
                           np.clip(p[:, 1].round().astype(int), 0, sel.shape[1] - 1)])
    H = max(1.3 * d0, 8 * s)
    off = np.arange(-H, H + 0.25, 0.5)
    yy = p[:, 0:1] + off[None] * nrm[:, 0:1]
    xx = p[:, 1:2] + off[None] * nrm[:, 1:2]
    P = ndi.map_coordinates(g, [yy, xx], order=1, mode="nearest")
    P = ndi.gaussian_filter1d(P, 1.0 * s, axis=0, mode="nearest")    # smooth along vessel

    n = len(P)
    k = max(2, int(0.15 * len(off)))
    dt = off[1] - off[0]
    idx = np.arange(len(off))
    # background: robust linear trend between the two profile ends
    bgL = np.median(P[:, :k], axis=1, keepdims=True)
    bgR = np.median(P[:, -k:], axis=1, keepdims=True)
    lam = (off - off[0]) / (off[-1] - off[0])
    BG = bgL + (bgR - bgL) * lam[None]
    AB = np.clip(-np.log((P + 1) / (BG + 1)), 0, None)       # Beer-Lambert absorbance
    dP = np.gradient(P, axis=1)
    core = np.abs(off) <= max(0.6 * d0, 3 * s)
    search = np.abs(off) <= 0.85 * H
    eL = np.zeros(n)
    eR = np.zeros(n)
    for i in range(n):                                      # pass 1: raw edges
        ci = np.flatnonzero(core)[np.argmax(AB[i, core])]
        lw = np.flatnonzero(search[:ci + 1])
        rw = ci + np.flatnonzero(search[ci:])
        li = lw[np.argmin(dP[i, lw])] if len(lw) else 0
        ri = rw[np.argmax(dP[i, rw])] if len(rw) else len(off) - 1
        eL[i] = np.interp(_subpix(-dP[i], li), idx, off)
        eR[i] = np.interp(_subpix(dP[i], ri), idx, off)
    # edges of a real vessel are continuous: median-filter along the centreline
    km = max(5, int(3 * s) * 2 + 1)
    eL, eR = medfilt(eL, km), medfilt(eR, km)
    D = np.maximum(eR - eL, 0.5)
    A = np.empty(n)
    for i in range(n):                                      # pass 2: densitometry
        wdw = (off >= eL[i] - 0.5 * D[i]) & (off <= eR[i] + 0.5 * D[i])
        A[i] = AB[i, wdw].sum() * dt
    m = max(3, int(2 * s) | 1)
    A = medfilt(A, m)

    inside = _in_box(p, bl)
    trim = int(max(2, H / 2))
    valid = np.zeros(n, bool)
    valid[trim:n - trim] = True
    inside &= valid
    if not inside.any():
        return Measurement(False, "centreline misses box")
    healthy = valid & ~inside

    Dref = _robust_reference(arc, D, healthy, valid)
    Aref = _robust_reference(arc, A, healthy, valid)
    rel_d = np.clip(D / Dref, 0, 1)
    rel_a = np.clip(A / Aref, 0, 1)
    rel_dd = np.sqrt(rel_a)                                   # densitometric diameter ratio

    rel = {"densitometric": rel_dd, "edge": rel_d}.get(primary, 0.5 * (rel_d + rel_dd))
    ii = np.flatnonzero(inside)
    i_mld = ii[np.argmin(rel[ii])]
    ds_edge = 100 * (1 - rel_d[ii].min())
    ds_dens = 100 * (1 - rel_dd[ii].min())
    as_dens = 100 * (1 - rel_a[ii].min())
    ds = 100 * (1 - rel[ii].min())

    ref = float(Dref[i_mld])
    mld = float(ref * rel[i_mld])

    narrow = rel < lesion_threshold
    lo = hi = i_mld
    while lo > 0 and narrow[lo - 1]:
        lo -= 1
    while hi < n - 1 and narrow[hi + 1]:
        hi += 1
    seg = slice(lo, hi + 1)
    length = float(arc[hi] - arc[lo]) if hi > lo else 1.0
    area = float(np.sum(Dref[seg] * (1 - rel[seg])))           # ds = 1 px
    vol = float(np.sum(np.pi / 4 * Dref[seg] ** 2 * (1 - rel_a[seg])))

    # quality: healthy support, agreement of both methods, contrast
    q = min(1.0, healthy.sum() / (4 * d0 + 1)) * np.exp(-abs(ds_edge - ds_dens) / 40)

    mld, ref, length, area, vol = mld / s, ref / s, length / s, area / s ** 2, vol / s ** 3
    mm = Measurement(True, "", mld, ref, ds, ds_edge, ds_dens, as_dens, length, area, vol)
    if mm_per_pixel:
        mm.mld_mm, mm.ref_diameter_mm = mld * mm_per_pixel, ref * mm_per_pixel
        mm.lesion_length_mm = length * mm_per_pixel
        mm.plaque_area_mm2 = area * mm_per_pixel ** 2
        mm.plaque_volume_mm3 = vol * mm_per_pixel ** 3
    mm.severity = severity_label(ds)
    mm.quality = float(q)
    sh = np.array([y0, x0])
    full = p / s + sh
    mm.centerline = full[:, ::-1].tolist()
    left = (p + eL[:, None] * nrm) / s + sh
    right = (p + eR[:, None] * nrm) / s + sh
    mm.edges = (left[:, ::-1].tolist(), right[:, ::-1].tolist())
    mm.lesion_span = (lo, hi)
    mm.mld_point = tuple(full[i_mld, ::-1])
    mm.profile = dict(arc=(arc / s).tolist(), D=(D / s).tolist(), Dref=(Dref / s).tolist(),
                      A=A.tolist(), Aref=Aref.tolist(), inside=inside.tolist())
    return mm


# ---------------------------------------------------------------- drawing
def draw(image, box, conf=None, m: Measurement | None = None, color=(0, 0, 255), show_edges=True):
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    x, y, w, h = map(int, box)
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    lines = []
    if conf is not None:
        lines.append(f"stenosis {conf:.2f}")
    if m is not None and m.ok:
        cl = np.int32(np.round(m.centerline))
        if show_edges:
            for e in m.edges:
                cv2.polylines(out, [np.int32(np.round(e))], False, (0, 200, 255), 1, cv2.LINE_AA)
        lo, hi = m.lesion_span
        cv2.polylines(out, [cl[lo:hi + 1]], False, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(out, tuple(int(v) for v in m.mld_point), 3, (255, 0, 255), -1)
        lines.append(f"DS {m.percent_diameter_stenosis:.0f}% {m.severity.split()[0]}")
        if np.isfinite(m.mld_mm):
            lines.append(f"MLD {m.mld_mm:.2f}mm L {m.lesion_length_mm:.1f}mm")
            lines.append(f"A {m.plaque_area_mm2:.1f}mm2 V~{m.plaque_volume_mm3:.1f}mm3")
        else:
            lines.append(f"MLD {m.mld_px:.1f}px L {m.lesion_length_px:.0f}px")
            lines.append(f"A {m.plaque_area_px2:.0f}px2 V~{m.plaque_volume_px3:.0f}px3")
    ty = y - 8 - 15 * (len(lines) - 1)
    if ty < 12:
        ty = y + h + 15
    for i, txt in enumerate(lines):
        pt = (max(2, x), ty + 15 * i)
        cv2.putText(out, txt, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out
