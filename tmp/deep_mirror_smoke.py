"""Smoke test (論点2): in a DEEP forward-only NNN, does the cov_weight mirror
develop a STRUCTURED error from correlated layer fluctuations, and can DECORRELATING
the fluctuations remove it?  (If yes -> start docs/idea_deep.md.)

cov_weight recovers W_ji = Cov_t(d_next_j, z_prev_i)/Var_t(z_prev_i), which equals
the true weight ONLY IF the z_prev_i fluctuate independently (nnn/credit.py). With
correlated fluctuations (deep layers share upstream noise), single-variable
regression suffers omitted-variable bias:
    W_hat_ji = W_ji + sum_{k!=i} W_jk Cov(z_k,z_i)/Var(z_i)
    => W_hat = W_true @ R,   R_ki = Cov(z_k,z_i)/Var(z_i)   (R=I iff independent)
So the mirror error = W_true @ (R - I): STRUCTURED by the activity correlation, and
removed by whitening (R->I). We verify this in a controlled linear mirror (A) and
measure the correlation that arises in a realistic multi-layer crossing net (B).
"""
import numpy as np


def cov_weight_np(d_next, z_prev, eps=1e-6):
    """W_hat[j,i] = mean_n Cov_t(d_j, z_i)/Var_t(z_i). d:[N,T,Ho] z:[N,T,Hi]."""
    cd = d_next - d_next.mean(1, keepdims=True)
    cz = z_prev - z_prev.mean(1, keepdims=True)
    cov = np.einsum('nto,nti->noi', cd, cz) / d_next.shape[1]
    var = (cz ** 2).mean(1)                                  # [N,Hi]
    return (cov / (var[:, None, :] + eps)).mean(0)


def corr_normalized_cov(z_prev, eps=1e-6):
    """R_ki = mean_n Cov_t(z_k,z_i)/Var_t(z_i) (the matrix s.t. W_hat = W_true@R)."""
    cz = z_prev - z_prev.mean(1, keepdims=True)
    cov = np.einsum('ntk,nti->nki', cz, cz) / z_prev.shape[1]
    var = (cz ** 2).mean(1)
    return (cov / (var[:, None, :] + eps)).mean(0)


def crossing(x, h=0.2):
    b1 = (x > h).astype(float); b2 = (x > -h).astype(float)
    return 0.5 * (np.abs(np.roll(b1, -1, 1) - b1) + np.abs(np.roll(b2, -1, 1) - b2))


def relerr(A, B):
    return np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-12)


# ---------- A: controlled linear mirror, known fluctuation correlation ----------
def part_A(N=8, T=4000, Hi=16, Ho=12, mix=0.7, seed=0):
    rng = np.random.default_rng(seed)
    W_true = rng.standard_normal((Ho, Hi)) / np.sqrt(Hi)
    # correlated fluctuations: F_indep @ C^T, C = (1-mix) I + mix * shared
    C = (1 - mix) * np.eye(Hi) + mix * rng.standard_normal((Hi, Hi)) / np.sqrt(Hi)
    print(f"=== A. controlled linear mirror (fluctuation mix={mix}) ===")
    for label, use_mix in (("independent fluct.", False), ("correlated fluct.", True)):
        F = rng.standard_normal((N, T, Hi))
        z = F @ C.T if use_mix else F
        z = z + rng.uniform(-1, 1, (N, 1, Hi))              # per-input signal (mean)
        d = np.einsum('nti,oi->nto', z, W_true)            # d = z W^T (linear pre-act)
        W_hat = cov_weight_np(d, z)
        R = corr_normalized_cov(z)
        pred = W_true @ R                                   # predicted biased mirror
        off = np.abs(R - np.eye(Hi))[~np.eye(Hi, dtype=bool)].mean()
        print(f"  {label:20s}: mirror relerr(W_hat,W_true)={relerr(W_hat, W_true):.3f}  "
              f"relerr(W_hat, W_true@R)={relerr(W_hat, pred):.3f}  mean|R-I|_offdiag={off:.3f}")
        if use_mix:
            # decorrelate (whiten) the fluctuations, then re-estimate the mirror
            cz = z - z.mean(1, keepdims=True)
            Sig = np.einsum('nti,ntj->ij', cz, cz) / (N * T)
            ev, U = np.linalg.eigh(Sig); Wh = U @ np.diag(1 / np.sqrt(ev + 1e-8)) @ U.T
            z_white = z.mean(1, keepdims=True) + cz @ Wh.T
            d_white = np.einsum('nti,oi->nto', z_white, W_true)
            W_hat_w = cov_weight_np(d_white, z_white)
            print(f"  {'-> after whitening':20s}: mirror relerr(W_hat,W_true)="
                  f"{relerr(W_hat_w, W_true):.3f}  (should drop toward 0)")


# ---------- B: realistic deep crossing net -- does correlation arise? ----------
def part_B(N=6, T=4000, H=24, depth=3, sigma=0.6, numT_note="", seed=0):
    rng = np.random.default_rng(seed)
    print(f"\n=== B. realistic {depth}-layer crossing net: fluctuation correlation "
          f"per layer ===")
    Ws = [rng.standard_normal((H, H)) / np.sqrt(H) for _ in range(depth)]
    x_in = rng.uniform(-1, 1, (N, 1, H)) + np.zeros((N, T, H))   # per-input signal
    z = x_in
    for l in range(depth):
        pre = np.einsum('nti,oi->nto', z, Ws[l]) + sigma * rng.standard_normal((N, T, H))
        z = crossing(pre)
        R = corr_normalized_cov(z)
        off = np.abs(R - np.eye(H))[~np.eye(H, dtype=bool)].mean()
        # mirror error at THIS layer's incoming weight (regress this pre on prev z)
        print(f"  layer {l+1}: mean|R-I|_offdiag(fluct. corr)={off:.3f}")


def main():
    part_A(mix=0.0)      # sanity: independent -> no error
    part_A(mix=0.7)      # correlated -> structured error + whitening fix
    part_B()
    print("\nVERDICT: premise holds if (A) correlated-fluct mirror error is large,"
          " matches W_true@R (structured), whitening removes it; and (B) deep layers"
          " develop non-trivial fluctuation correlation.")


if __name__ == "__main__":
    main()
