# -*- coding: utf-8 -*-
"""Independently verify the released split: no acquisition sequence and no
near-duplicate group may span two subsets."""
import os, re, argparse, datetime, collections
import numpy as np
from PIL import Image

_ap = argparse.ArgumentParser(
    description="Independently verify that no acquisition sequence and no near-duplicate "
                "group spans two subsets of the released split.")
_ap.add_argument("--images", default="data/images",
                 help="directory containing the image files (default: data/images)")
_ap.add_argument("--labels", default="data/labels",
                 help="directory containing the YOLO label files (default: data/labels)")
_ap.add_argument("--splits", default="splits",
                 help="directory containing train.txt / val.txt / test.txt (default: splits)")
_args = _ap.parse_args()

IMG_DIR, LBL_DIR, SPL = _args.images, _args.labels, _args.splits

def family(f):
    s = os.path.splitext(f)[0]
    if re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2}\.\d+', s): return "VID_A"
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}\.\d+', s): return "VID_B"
    if re.fullmatch(r'IMG_\d+', s): return "IMG"
    if re.fullmatch(r'\d+', s): return "NUM"
    return "OTHER"

def parse_ts(f):
    s = os.path.splitext(f)[0]
    m = re.match(r'(\d{4}-\d{2}-\d{2})[ _](\d{2})_(\d{2})_(\d{2})\.(\d+)', s)
    d, hh, mm, ss, mic = m.groups()
    return datetime.datetime.strptime(f"{d} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S") + \
           datetime.timedelta(microseconds=int(mic))

where = {}
for s in ("train", "val", "test"):
    with open(os.path.join(SPL, s + ".txt"), encoding="utf-8") as fh:
        for l in fh:
            if l.strip(): where[l.strip()] = s
print("images mapped:", len(where))

# 1. acquisition sequences (video clips) must not span subsets
bad_clip = 0; clips = 0
for famname in ("VID_A", "VID_B"):
    fs = sorted([f for f in where if family(f) == famname], key=parse_ts)
    cur = [fs[0]]
    for i in range(len(fs) - 1):
        if (parse_ts(fs[i+1]) - parse_ts(fs[i])).total_seconds() > 5:
            if len({where[x] for x in cur}) > 1: bad_clip += 1
            clips += 1; cur = []
        cur.append(fs[i+1])
    if len({where[x] for x in cur}) > 1: bad_clip += 1
    clips += 1
print(f"video clips checked        : {clips}   clips spanning two subsets: {bad_clip}")

# 2. near-duplicate components (dHash <= 4, within family) must not span subsets
def dhash(f, hs=8):
    with Image.open(os.path.join(IMG_DIR, f)) as im:
        g = im.convert("L").resize((hs+1, hs), Image.BILINEAR)
        a = np.asarray(g, dtype=np.int16)
        return (a[:, 1:] > a[:, :-1]).flatten()

bad_dup = 0; comps = 0; dup_sizes = []
for famname in ("VID_A", "VID_B", "IMG", "NUM", "OTHER"):
    fs = [f for f in where if family(f) == famname]
    if not fs: continue
    H = np.array([dhash(f) for f in fs])
    par = list(range(len(fs)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(len(fs)):
        d = np.count_nonzero(H[i+1:] != H[i], axis=1)
        for j in np.nonzero(d <= 4)[0]:
            a, b = find(i), find(i+1+int(j))
            if a != b: par[b] = a
    grp = collections.defaultdict(list)
    for i in range(len(fs)): grp[find(i)].append(fs[i])
    for g in grp.values():
        comps += 1
        if len(g) > 1: dup_sizes.append(len(g))
        if len({where[x] for x in g}) > 1: bad_dup += 1
print(f"near-dup components checked: {comps}   components spanning two subsets: {bad_dup}")
print(f"  multi-image components   : {len(dup_sizes)}   largest: {max(dup_sizes) if dup_sizes else 0}")

# 3. cross-check the published per-split instance counts
LBL = LBL_DIR
print(f"\n{'subset':8s} {'images':>7s} {'dust':>7s} {'bird':>7s} {'total':>7s}")
for s in ("train", "val", "test"):
    fs = [f for f in where if where[f] == s]
    d = b = 0
    for f in fs:
        lp = os.path.join(LBL, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lp):
            with open(lp) as fh:
                for line in fh:
                    t = line.split()
                    if len(t) != 5: continue
                    if int(float(t[0])) == 0: d += 1
                    else: b += 1
    print(f"{s:8s} {len(fs):7d} {d:7d} {b:7d} {d+b:7d}")
print("\nVERDICT:", "PASS - no leakage" if bad_clip == 0 and bad_dup == 0 else "FAIL")
