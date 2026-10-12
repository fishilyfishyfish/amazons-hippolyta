"""Trains the evaluation network on self-play data (engine/examples/selfplay.rs).

Pure numpy (no deep-learning framework is available in the build container):
convolutions are im2col + matmul, gradients are written by hand and checked
numerically by `--gradcheck`.

    python3 tools/nn/train.py --data a.bin [b.bin ...] --channels 32 --blocks 2 \
        --epochs 6 --out web/net.bin [--dump parity.bin]

Record layout, input planes and the network must match engine/src/nn.rs.
Activations are kept channels-last: (batch, 100 squares, channels).
"""
import argparse
import struct
import time

import numpy as np

PLANES = 13
MAX_CANDS = 40
RECORD = PLANES * 16 + 1 + MAX_CANDS * 5 + 4 + 1
F = np.float32

# ---------------------------------------------------------------- data


def load(paths):
    raw = np.concatenate([np.fromfile(p, dtype=np.uint8) for p in paths])
    n = len(raw) // RECORD
    raw = raw[: n * RECORD].reshape(n, RECORD)
    pb = raw[:, : PLANES * 16].reshape(n, PLANES, 16)
    bits = np.unpackbits(pb, axis=2, bitorder="little")[:, :, :100]  # (n, 13, 100)
    x = np.ascontiguousarray(bits.transpose(0, 2, 1)).astype(F)  # (n, 100, 13)
    off = PLANES * 16
    ncand = raw[:, off].astype(np.int64)
    c = raw[:, off + 1 : off + 1 + MAX_CANDS * 5].reshape(n, MAX_CANDS, 5)
    idx = [c[:, :, k].astype(np.int64) for k in range(3)]
    visits = c[:, :, 3].astype(F) + 256 * c[:, :, 4].astype(F)
    visits *= np.arange(MAX_CANDS)[None, :] < ncand[:, None]
    tot = visits.sum(1, keepdims=True)
    tot[tot == 0] = 1
    w = visits / tot
    pol = np.zeros((n, 3, 100), F)
    rows = np.repeat(np.arange(n), MAX_CANDS)
    for k in range(3):
        np.add.at(pol[:, k], (rows, idx[k].reshape(-1)), w.reshape(-1))
    off2 = off + 1 + MAX_CANDS * 5
    q = raw[:, off2 : off2 + 4].copy().view(np.float32)[:, 0]
    z = raw[:, off2 + 4].astype(np.int8).astype(F)
    return x, pol, q, z


MIRROR = np.array([r * 10 + (9 - c) for r in range(10) for c in range(10)])

# Neighbour index for 3x3 kernels: NB[sq, k] = square at offset k (k =
# (dr+1)*3 + (dc+1), same as engine/src/nn.rs), or 100 (a zero pad) off-board.
NB = np.full((100, 9), 100, np.int64)
for sq in range(100):
    for k in range(9):
        r, cc = sq // 10 + k // 3 - 1, sq % 10 + k % 3 - 1
        if 0 <= r < 10 and 0 <= cc < 10:
            NB[sq, k] = r * 10 + cc

# ---------------------------------------------------------------- layers


def conv3_fwd(a, w, b):
    """a (n,100,ci), w (9*ci, co) -> out (n,100,co), cols cache."""
    n, _, ci = a.shape
    ap = np.concatenate([a, np.zeros((n, 1, ci), F)], axis=1)
    cols = ap[:, NB, :].reshape(n * 100, 9 * ci)  # (k, ci) order
    return (cols @ w + b).reshape(n, 100, -1), cols


def conv3_bwd(dout, cols, w, ci):
    n = dout.shape[0]
    d2 = dout.reshape(n * 100, -1)
    dw = cols.T @ d2
    db = d2.sum(0)
    dcols = (d2 @ w.T).reshape(n, 100, 9, ci)
    dap = np.zeros((n, 101, ci), F)
    for k in range(9):
        dap[:, NB[:, k], :] += dcols[:, :, k, :]  # one-to-one per k (pad discarded)
    return dap[:, :100, :], dw, db


class Model:
    def __init__(self, c, blocks, rng):
        def he(fan_in, shape):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(F)

        self.c, self.blocks = c, blocks
        p = {}
        p["stem_w"] = he(9 * PLANES, (9 * PLANES, c)); p["stem_b"] = np.zeros(c, F)
        for i in range(blocks):
            p[f"r{i}a_w"] = he(9 * c, (9 * c, c)); p[f"r{i}a_b"] = np.zeros(c, F)
            p[f"r{i}b_w"] = he(9 * c, (9 * c, c)) * 0.5; p[f"r{i}b_b"] = np.zeros(c, F)
        p["v_w"] = he(c, (c, 2)); p["v_b"] = np.zeros(2, F)
        p["fc1_w"] = he(200, (200, 64)); p["fc1_b"] = np.zeros(64, F)
        p["fc2_w"] = (rng.standard_normal((64, 1)) * 0.05).astype(F); p["fc2_b"] = np.zeros(1, F)
        p["p_w"] = (rng.standard_normal((c, 3)) * 0.05).astype(F); p["p_b"] = np.zeros(3, F)
        self.p = p

    def forward(self, x):
        p, cache = self.p, {}
        h, cache["stem"] = conv3_fwd(x, p["stem_w"], p["stem_b"])
        cache["stem_pre"] = h
        h = np.maximum(h, 0)
        for i in range(self.blocks):
            t, ca = conv3_fwd(h, p[f"r{i}a_w"], p[f"r{i}a_b"])
            ta = np.maximum(t, 0)
            u, cb = conv3_fwd(ta, p[f"r{i}b_w"], p[f"r{i}b_b"])
            pre = h + u
            cache[f"r{i}"] = (h, t, ta, ca, cb, pre)
            h = np.maximum(pre, 0)
        cache["trunk"] = h
        n = x.shape[0]
        vpre = h @ p["v_w"] + p["v_b"]  # (n,100,2)
        v = np.maximum(vpre, 0)
        vflat = v.transpose(0, 2, 1).reshape(n, 200)  # [ch][sq] like Rust
        f1pre = vflat @ p["fc1_w"] + p["fc1_b"]
        f1 = np.maximum(f1pre, 0)
        out = np.tanh((f1 @ p["fc2_w"] + p["fc2_b"])[:, 0])
        logits = (h @ p["p_w"] + p["p_b"]).transpose(0, 2, 1)  # (n,3,100)
        cache.update(vpre=vpre, vflat=vflat, f1pre=f1pre, f1=f1, out=out)
        return out, logits, cache

    def backward(self, cache, dout, dlogits):
        p, g = self.p, {}
        n = dout.shape[0]
        h = cache["trunk"]
        # value head
        dz = dout * (1 - cache["out"] ** 2)
        g["fc2_w"] = cache["f1"].T @ dz[:, None]
        g["fc2_b"] = dz.sum(keepdims=True)
        df1 = dz[:, None] @ p["fc2_w"].T * (cache["f1pre"] > 0)
        g["fc1_w"] = cache["vflat"].T @ df1
        g["fc1_b"] = df1.sum(0)
        dvflat = df1 @ p["fc1_w"].T
        dv = dvflat.reshape(n, 2, 100).transpose(0, 2, 1) * (cache["vpre"] > 0)
        g["v_w"] = h.reshape(-1, self.c).T @ dv.reshape(-1, 2)
        g["v_b"] = dv.reshape(-1, 2).sum(0)
        dh = dv @ p["v_w"].T
        # policy head
        dl = dlogits.transpose(0, 2, 1)  # (n,100,3)
        g["p_w"] = h.reshape(-1, self.c).T @ dl.reshape(-1, 3)
        g["p_b"] = dl.reshape(-1, 3).sum(0)
        dh = dh + dl @ p["p_w"].T
        # trunk
        for i in reversed(range(self.blocks)):
            hin, t, ta, ca, cb, pre = cache[f"r{i}"]
            dpre = dh * (pre > 0)
            dta, g[f"r{i}b_w"], g[f"r{i}b_b"] = conv3_bwd(dpre, cb, p[f"r{i}b_w"], self.c)
            dt = dta * (t > 0)
            dhin, g[f"r{i}a_w"], g[f"r{i}a_b"] = conv3_bwd(dt, ca, p[f"r{i}a_w"], self.c)
            dh = dpre + dhin
        dh = dh * (cache["stem_pre"] > 0)
        _, g["stem_w"], g["stem_b"] = conv3_bwd(dh, cache["stem"], p["stem_w"], PLANES)
        return g


def loss_and_grads(model, x, vt, pol, pw=1.0 / 3):
    n = x.shape[0]
    out, logits, cache = model.forward(x)
    lv = np.mean((out - vt) ** 2)
    m = logits.max(2, keepdims=True)
    e = np.exp(logits - m)
    sm = e / e.sum(2, keepdims=True)
    logsm = logits - m - np.log(e.sum(2, keepdims=True))
    lp = -(pol * logsm).sum(2).sum(1).mean()
    dout = (2 * (out - vt) / n).astype(F)
    dlogits = (pw * (sm * pol.sum(2, keepdims=True) - pol) / n).astype(F)
    g = model.backward(cache, dout, dlogits)
    return lv, lp, g, out, logits


class Adam:
    def __init__(self, params, lr, wd=1e-4):
        self.lr, self.wd, self.t = lr, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr):
        self.t += 1
        b1, b2 = 0.9, 0.999
        for k in params:
            gk = grads[k].astype(F)
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            params[k] -= (lr * (mh / (np.sqrt(vh) + 1e-8) + self.wd * params[k])).astype(F)


def export(model, path):
    p, c = model.p, model.c
    parts = [b"AMNN", struct.pack("<III", 1, c, model.blocks)]

    def conv3(w, ci):  # (9*ci, co) in (k, ci) order -> [co][ci][9]
        return w.reshape(9, ci, -1).transpose(2, 1, 0)

    def add(a):
        parts.append(np.ascontiguousarray(a, dtype="<f4").tobytes())

    add(conv3(p["stem_w"], PLANES)); add(p["stem_b"])
    for i in range(model.blocks):
        add(conv3(p[f"r{i}a_w"], c)); add(p[f"r{i}a_b"])
        add(conv3(p[f"r{i}b_w"], c)); add(p[f"r{i}b_b"])
    add(p["v_w"].T); add(p["v_b"])
    add(p["fc1_w"].T); add(p["fc1_b"])
    add(p["fc2_w"].T); add(p["fc2_b"])
    add(p["p_w"].T); add(p["p_b"])
    with open(path, "wb") as f:
        f.write(b"".join(parts))


def gradcheck():
    rng = np.random.default_rng(1)
    model = Model(4, 1, rng)
    for k in model.p:
        model.p[k] = model.p[k].astype(np.float64)
        if k.endswith("_b"):
            # Zero biases put many ReLUs exactly on their kink, where finite
            # differences are meaningless.
            model.p[k] += rng.uniform(-0.3, 0.3, model.p[k].shape)
    global F
    F = np.float64
    x = (rng.random((3, 100, PLANES)) < 0.3).astype(np.float64)
    vt = rng.uniform(-1, 1, 3)
    pol = rng.random((3, 3, 100)); pol /= pol.sum(2, keepdims=True)

    def total():
        lv, lp, _, _, _ = loss_and_grads(model, x, vt, pol)
        return lv + lp / 3

    _, _, g, _, _ = loss_and_grads(model, x, vt, pol)
    worst = 0.0
    for k, v in model.p.items():
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in v.shape)
            old = v[i]
            v[i] = old + 1e-5; a = total()
            v[i] = old - 1e-5; b = total()
            v[i] = old
            num = (a - b) / 2e-5
            err = abs(num - g[k][i]) / max(1e-6, abs(num) + abs(g[k][i]))
            worst = max(worst, err)
            if err > 1e-4:
                print(f"MISMATCH {k}{i}: numeric {num:.6g} analytic {g[k][i]:.6g}")
    print(f"gradcheck worst relative error {worst:.2e}")
    return worst < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--qmix", type=float, default=0.5, help="value target = qmix*q + (1-qmix)*z")
    ap.add_argument("--out")
    ap.add_argument("--dump", help="sample inputs/outputs for the Rust parity test")
    ap.add_argument("--gradcheck", action="store_true")
    args = ap.parse_args()
    if args.gradcheck:
        raise SystemExit(0 if gradcheck() else 1)

    rng = np.random.default_rng(0)
    x, pol, q, z = load(args.data)
    vt = (args.qmix * q + (1 - args.qmix) * z).astype(F)
    # Left-right mirror augmentation.
    x = np.concatenate([x, x[:, MIRROR, :]])
    pol = np.concatenate([pol, pol[:, :, MIRROR]])
    vt = np.concatenate([vt, vt])
    n = len(x)
    perm = rng.permutation(n)
    nval = max(1000, n // 20)
    val, tr = perm[:nval], perm[nval:]
    print(f"{n} samples ({n // 2} positions x2 mirror), {nval} for validation", flush=True)

    model = Model(args.channels, args.blocks, rng)
    opt = Adam(model.p, args.lr)
    steps = args.epochs * (len(tr) // args.batch)
    step = 0

    def evaluate():
        lvs, lps, accs = [], [], []
        for i in range(0, nval, 1024):
            idx = val[i : i + 1024]
            out, logits, _ = model.forward(x[idx])
            lvs.append(np.mean((out - vt[idx]) ** 2) * len(idx))
            m = logits.max(2, keepdims=True)
            logsm = logits - m - np.log(np.exp(logits - m).sum(2, keepdims=True))
            lps.append(-(pol[idx] * logsm).sum(2).sum(1).sum())
            accs.append((logits[:, 0].argmax(1) == pol[idx, 0].argmax(1)).sum())
        return sum(lvs) / nval, sum(lps) / nval, sum(accs) / nval

    lv, lp, acc = evaluate()
    print(f"start: val value mse {lv:.4f}, policy ce {lp:.3f}, from top1 {acc:.3f}", flush=True)
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(tr)
        for i in range(0, len(tr) - args.batch + 1, args.batch):
            idx = tr[i : i + args.batch]
            # one-cycle-ish: warm up 5%, cosine decay
            frac = step / max(1, steps)
            lr = args.lr * (frac / 0.05 if frac < 0.05 else 0.5 * (1 + np.cos(np.pi * (frac - 0.05) / 0.95)))
            _, _, g, _, _ = loss_and_grads(model, x[idx], vt[idx], pol[idx])
            opt.step(model.p, g, lr)
            step += 1
        lv, lp, acc = evaluate()
        print(f"epoch {ep + 1}: val value mse {lv:.4f}, policy ce {lp:.3f}, from top1 {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    export(model, args.out)
    print("wrote", args.out)
    if args.dump:
        idx = val[:8]
        out, logits, _ = model.forward(x[idx])
        with open(args.dump, "wb") as f:
            f.write(struct.pack("<I", len(idx)))
            # inputs as [n][plane][sq] like dense_planes()
            f.write(np.ascontiguousarray(x[idx].transpose(0, 2, 1), dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(out, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(logits, dtype="<f4").tobytes())


if __name__ == "__main__":
    main()
