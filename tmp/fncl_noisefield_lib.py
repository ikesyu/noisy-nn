"""
fncl_noisefield_lib.py — ノイズ場（noise field）実験の共通部（docs/idea_ca.md §5）
（旧名: fncl_nf.py．出力ディレクトリ tmp/out/ は旧名のまま）

層内相関の抑制を狙うノイズ場の設計と測定を担う。中核は「周期ノイズ層」と
「位相内中心化」の 2 つ。

## なぜ周期ノイズなのか（§5.5）

交差活性 `nnn/activation.py::_cyclic_xor` は T 方向の**隣接サンプル間**の XOR であり、
定義上の 2 つの iid ノイズは連続サンプルとして実現されている:

    z_t = |1{d_{t+1} + eta_{t+1} > h} - 1{d_t + eta_t > h}|

したがって連続区間でノイズを固定する素朴なブロック凍結は使えない（隣接が同一に
なり XOR が恒等的に 0、ユニットが死ぬ）。代わりに**周期 p でタイルする**:

    eta_t = eta_{t mod p}      (p >= 2, p | T)

隣接サンプルは異なるので交差活性は正常に発火し、出力 z も周期 p になる。
上流層をこうしておくと、位相クラス G_c = {t : t = c mod p} の中では
d が定数対 (d_c, d_{c+1}) に固定され、対象層（周期 1）の z の変動は
そのユニット固有のノイズだけに由来する = ユニット間で構成上独立になる。

よって位相内で中心化して共分散を取れば、単変量ミラーは不偏になる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parent.parent / "data_nce"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))
import fncl  # noqa: E402
from nnn.activation import CrossingSample as _CrossingSample  # noqa: E402

EPS = 1e-8


# ============================================================
# 周期ノイズ層（ノイズ場の時間構造）
# ============================================================
class PeriodicGaussianNoise(nn.Module):
    """周期 p でタイルしたガウスノイズを加える（p=1 なら通常の i.i.d.）。

    nnn.layer.GaussianNoiseLayer と同じインタフェース（forward(x, std=None)）を
    持つので、`crossing_layer.noise_layer` をそのまま差し替えられる。
    std はスカラーでも [D] のノイズ場ベクトルでもよい。
    """

    def __init__(self, std, period: int = 1, mean: float = 0.0,
                 scale: float = 1.0):
        super().__init__()
        self.std = std
        self.mean = mean
        self.period = int(period)
        # モデルの forward は毎回 crossing 層の std を渡してくるので、
        # ノイズ場としての倍率は scale として別に保持し、常に掛ける（L1 用）。
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, std=None) -> torch.Tensor:
        if std is not None:
            self.std = std
        n, t, d = x.shape
        p = self.period
        if p <= 1:
            eps = torch.randn_like(x)
        else:
            if t % p != 0:
                raise ValueError(f"T={t} must be divisible by period={p}")
            base = torch.randn(n, p, d, device=x.device, dtype=x.dtype)
            eps = base.repeat(1, t // p, 1)      # eta_t = eta_{t mod p}（タイル）
        return x + eps * (self.std * self.scale) + self.mean


class PeriodicUniformCrossing(nn.Module):
    """一様ノイズ版の周期ノイズ付き交差活性（nnn.layer.UniformCrossingSampleLayer 互換）。

    一様版の交差層は noise_layer を持たず forward 内でノイズを生成するので、
    ガウス版のように noise_layer を差し替えることができない。モジュールごと置き換える。
    周期化の考え方（eta_t = eta_{t mod p}）はガウス版と同一。
    """

    def __init__(self, radius, center: float = 0.0, h: float = 0.2,
                 period: int = 1, scale: float = 1.0):
        super().__init__()
        self.radius = radius
        self.center = center
        self.h = h
        self.period = int(period)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, radius=None) -> torch.Tensor:
        r = (radius if radius is not None else self.radius)
        n, t, d = x.shape
        p = self.period
        if p <= 1:
            u = torch.rand_like(x)
        else:
            if t % p != 0:
                raise ValueError(f"T={t} must be divisible by period={p}")
            u = torch.rand(n, p, d, device=x.device, dtype=x.dtype).repeat(1, t // p, 1)
        eps = self.center + (r * self.scale) * (2.0 * u - 1.0)
        return _CrossingSample.apply(x + eps, self.h)


def install_noise_field(net, periods=None, std_scale=None, sigma: float = None):
    """各隠れ層の noise_layer を周期ノイズ層に差し替える（ノイズ場の設定）。

    periods   : 層ごとの更新周期 [p_1, ..., p_L]（既定は全て 1 = 通常の NNN）
    std_scale : 層ごとの std 倍率（L1: 私的分散を増やす介入）
    sigma     : 基準 std（None なら現在の層の std を使う）
    元の noise_layer は net._nf_orig に退避し、restore_noise_field() で戻せる。
    """
    crossings = fncl.crossing_layers(net)
    n_hidden = len(crossings)
    if periods is None:
        periods = [1] * n_hidden
    if std_scale is None:
        std_scale = [1.0] * n_hidden
    uniform = not hasattr(crossings[0], "noise_layer")
    if not hasattr(net, "_nf_orig"):
        # ガウス版は noise_layer を，一様版は交差層モジュールそのものを退避する
        net._nf_orig = [c if uniform else c.noise_layer for c in crossings]
        net._nf_uniform = uniform
    for l, c in enumerate(crossings):
        if uniform:
            o = net._nf_orig[l]
            crossings[l] = PeriodicUniformCrossing(
                sigma if sigma is not None else o.radius, o.center, o.h,
                periods[l], scale=std_scale[l])
        else:
            base = sigma if sigma is not None else net._nf_orig[l].std
            c.noise_layer = PeriodicGaussianNoise(base, periods[l],
                                                  scale=std_scale[l])
    net._nf_periods = list(periods)
    return net


def restore_noise_field(net):
    if hasattr(net, "_nf_orig"):
        crossings = fncl.crossing_layers(net)
        for l, orig in enumerate(net._nf_orig):
            if getattr(net, "_nf_uniform", False):
                crossings[l] = orig
            else:
                crossings[l].noise_layer = orig
        net._nf_periods = [1] * len(net._nf_orig)
    return net


# ============================================================
# 位相内中心化
# ============================================================
def build_net_noise(noise: str, depth: int, width: int, scale: float, h: float,
                    t: int, device):
    """784-width*depth-10 の NNN を，ガウス／一様ノイズのどちらでも構築する。

    scale はガウスなら std，一様なら radius。fncl_mnist_fidelity.build_net の一般化版。
    """
    from nnn import model as nnn_model
    structure = [784] + [width] * depth + [10]
    if noise == "gaussian":
        net = nnn_model.SimpleNNNSample(structure=structure, std=scale, h=h, t=t,
                                        output_bias=True)
    elif noise == "uniform":
        net = nnn_model.SimpleNNNUniformSample(structure=structure, radius=scale,
                                               center=0.0, h=h, t=t,
                                               output_bias=True)
    else:
        raise ValueError(noise)
    return net.to(device)


def phase_center(a: torch.Tensor, period: int, pass_len: int = 0) -> torch.Tensor:
    """[N, T, D] を位相クラス（t mod period）ごとに中心化して返す。

    t = q * period + c と分解し、各 c について q 方向の平均を引く。
    period=1 なら通常の T 方向中心化に一致する。

    `pass_len` は 1 回の forward が生む T の長さ。複数パスを T 軸に連結している
    場合、パスごとに独立な周期ノイズが引かれているので、位相クラスは
    **パス境界を跨いではならない**（跨ぐと「上流が定数」という前提が壊れる）。
    pass_len を与えると (パス, 位相) の組ごとに中心化する。0 なら全 T を 1 パスとみなす。
    """
    n, t, d = a.shape
    if period <= 1:
        return a - a.mean(dim=1, keepdim=True)
    pl = pass_len if pass_len and pass_len < t else t
    if t % pl != 0 or pl % period != 0:
        raise ValueError(f"T={t}, pass_len={pl}, period={period} が割り切れない")
    v = a.view(n, t // pl, pl // period, period, d)      # [N, pass, q, c, D]
    return (v - v.mean(dim=2, keepdim=True)).view(n, t, d)


def cov_weight_phase(d_next: torch.Tensor, z_prev: torch.Tensor,
                     period: int, pass_len: int = 0) -> torch.Tensor:
    """位相内中心化版の単変量ミラー（fncl.cov_weight の pool=True と同形）。

    上流層が周期 period でタイルされているとき、位相クラス内では上流が定数なので
    z_prev のユニットは構成上独立になり、この単変量回帰は不偏になる（§5.5）。
    """
    cd = phase_center(d_next, period, pass_len)
    cz = phase_center(z_prev, period, pass_len)
    t = d_next.shape[1]
    cov = torch.einsum("nto,nti->oi", cd, cz) / t
    var = (cz ** 2).mean(dim=1)                              # [N, Hi]
    return cov / (var.sum(dim=0).unsqueeze(0) + EPS)


# ============================================================
# 共通モード除去（S0 の発見: 層内相関はランク 1 の一様な共通モードだった）
# ============================================================
def remove_common_mode(z: torch.Tensor, centered: bool = False) -> torch.Tensor:
    """各サンプルについて層内の全ユニット平均を引く（一様共通モードの除去）。

    S0 の測定では層内相関はほぼ一様（rho_ki ~ c，行和が |rho| の 255 倍）であり，
    これは「上流層の総発火数の揺らぎが下流の全ユニットを同じ向きに駆動する」ことに
    対応する。その共通成分は各サンプルでのユニット平均そのものなので，
    引き算 1 回（アキュムレータ 1 本，O(H)）で除去できる。
    行列の逆（O(H^3)）も，ノイズ更新スケジュールの制御も要らない。
    """
    a = z if centered else z - z.mean(dim=1, keepdim=True)
    return a - a.mean(dim=2, keepdim=True)          # ユニット方向の平均を引く


def remove_rank1(z: torch.Tensor, centered: bool = False) -> torch.Tensor:
    """先頭主成分（入力プールで推定）を射影除去する一般化版（ランク 1）。

    共通モードの負荷が一様でない場合（a_i が定数でない場合）に対応する。
    O(H) の内積 2 回で済み、逆行列は要らない。
    """
    a = z if centered else z - z.mean(dim=1, keepdim=True)
    n, t, d = a.shape
    flat = a.reshape(-1, d)
    # 冪乗法数回で先頭固有ベクトルを取る（H x H を作らずに済む）
    v = torch.randn(d, device=a.device, dtype=a.dtype)
    v /= v.norm() + EPS
    for _ in range(8):
        v = flat.T @ (flat @ v)
        v /= v.norm() + EPS
    coeff = flat @ v                                  # [n*t]
    return (flat - coeff.unsqueeze(1) * v.unsqueeze(0)).view(n, t, d)


def _common_mode_loadings(cz: torch.Tensor, var: torch.Tensor, uniform: bool,
                          iters: int = 6):
    """層内共通モードの負荷 s を leave-one-out 共分散から推定する（O(H)）。

    モデル: z_i = s_i f + e_i（f は層共通のスカラー因子, Var(f)=1, e_i は私的）
        Cov(z_i, z_k) = s_i s_k  (i != k),      Var(z_i) = s_i^2 + v_i

    集団活動 u = sum_k z_k との共分散は自己項を含む:
        Cov(z_i, u) = Var(z_i) + sum_{k!=i} s_i s_k
    したがって **Var(z_i) を引いてから** 使わなければならない:
        b_i := Cov(z_i, u) - Var(z_i) = s_i (S - s_i),   S = sum_k s_k
    これを引かずに Cov(z_i, u) をそのまま負荷とみなすと，共通モードが弱いときは
    対角成分 Var(z_i) に汚染されて s の方向を向かない（旧実装の誤り）。

    uniform=True なら s_i = s（一様負荷）とし，b の平均から s^2 = mean(b)/(H-1) と置く。
    uniform=False なら s_i = b_i / (S - s_i) の不動点反復で不均一な負荷を許す。
    """
    n, t, h = cz.shape
    u = cz.sum(dim=2, keepdim=True)                                  # [N, T, 1]
    cov_zu = (cz * u).mean(dim=1).mean(dim=0)                        # [H]
    b = (cov_zu - var).clamp_min(0.0)                                # s_i (S - s_i)
    s2 = float(b.mean()) / max(1, h - 1)
    s = torch.full_like(var, max(s2, 0.0) ** 0.5)
    if not uniform:
        s_max = torch.sqrt(var.clamp_min(0.0))
        for _ in range(iters):
            s = (b / (s.sum() - s).clamp_min(1e-8)).clamp(min=0.0)
            s = torch.minimum(s, s_max)
    return s


def cov_weight_rank1(d_next: torch.Tensor, z_prev: torch.Tensor,
                     uniform: bool = True, ridge: float = 1e-3) -> torch.Tensor:
    """一様（またはランク 1）共通モードを Sherman-Morrison で正しく除いたミラー。

    Cov(z,z) ~ diag(v) + s s^T と見て
        (diag(v) + s s^T)^{-1} = Dinv - Dinv s s^T Dinv / (1 + s^T Dinv s)
    を右から掛ける。行列を作らないので O(H*Ho)。負荷 s は自己項を除いた
    leave-one-out 共分散から推定する（_common_mode_loadings）。
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)
    t, n = d_next.shape[1], d_next.shape[0]
    cov_dz = torch.einsum("nto,nti->oi", cd, cz) / (t * n)
    var = (cz ** 2).mean(dim=1).mean(dim=0)                          # [Hi]
    s = _common_mode_loadings(cz, var, uniform)
    v = (var - s ** 2).clamp_min(ridge * float(var.mean()))
    dinv = 1.0 / v
    denom = 1.0 + float((s * dinv * s).sum())
    a = cov_dz * dinv.unsqueeze(0)
    return a - (a @ s).unsqueeze(1) * (s * dinv).unsqueeze(0) / denom


def cov_weight_woodbury(d_next: torch.Tensor, z_prev: torch.Tensor,
                        ridge: float = 1e-3) -> torch.Tensor:
    """L5(k=1): 層内共分散を diag + ランク 1 と見て Woodbury で逆にする（O(H)）。

    S0 より Cov(z,z) ~ diag(v) + g g^T（g = 集団活動との共分散 = 一様共通モード）。
    Woodbury:
        (D + g g^T)^{-1} = D^{-1} - D^{-1} g g^T D^{-1} / (1 + g^T D^{-1} g)
    なので W_hat = Cov(d,z) (D + g g^T)^{-1} は行列の逆を作らずに O(H*Ho) で計算できる。

    注意: 回帰子の平均を引くだけ（remove_common_mode）では**バイアスは消えない**。
    分子には c*s_i*sum_k W_jk s_k という項が残るからで、除くべきは回帰子の平均ではなく
    共分散行列の逆である。この関数がその厳密な（ランク 1 近似のもとでの）実装。
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)
    cz = z_prev - z_prev.mean(dim=1, keepdim=True)
    t, n = d_next.shape[1], d_next.shape[0]
    cov_dz = torch.einsum("nto,nti->oi", cd, cz) / (t * n)        # [Ho, Hi]
    v = (cz ** 2).mean(dim=1).mean(dim=0)                          # [Hi] 対角
    # 集団活動 u = sum_k z_k との共分散から共通モードの負荷 g を推定する（O(H)）
    u = cz.sum(dim=2, keepdim=True)                                # [N, T, 1]
    cov_zu = (cz * u).mean(dim=1).mean(dim=0)                      # [Hi]
    var_u = (u.squeeze(-1) ** 2).mean(dim=1).mean(dim=0)           # スカラー
    g = cov_zu / torch.sqrt(var_u.clamp_min(EPS))                  # [Hi]
    d_diag = (v - g ** 2).clamp_min(ridge * float(v.mean()))       # 私的分散
    dinv = 1.0 / d_diag
    denom = 1.0 + float((g * dinv * g).sum())
    a = cov_dz * dinv.unsqueeze(0)                                 # Cov(d,z) D^-1
    corr = (a @ g).unsqueeze(1) * (g * dinv).unsqueeze(0) / denom
    return a - corr


def cov_weight_decorr(d_next: torch.Tensor, z_prev: torch.Tensor,
                      mode: str = "cm") -> torch.Tensor:
    """共通モードを除いた回帰子で単変量ミラーを測る。

    mode: "cm" = ユニット平均の除去，"rank1" = 先頭主成分の射影除去。
    分子の共分散だけを共通モード除去後の回帰子で取り，分母は元の Var(z_i) を使う
    （回帰係数のスケールを保つため）。
    """
    cd = d_next - d_next.mean(dim=1, keepdim=True)
    cz_raw = z_prev - z_prev.mean(dim=1, keepdim=True)
    cz = (remove_common_mode(cz_raw, centered=True) if mode == "cm"
          else remove_rank1(cz_raw, centered=True))
    t = d_next.shape[1]
    cov = torch.einsum("nto,nti->oi", cd, cz) / t
    var = (cz * cz_raw).mean(dim=1)      # 射影後の回帰子と元の z の共分散＝有効分散
    return cov / (var.sum(dim=0).unsqueeze(0) + EPS)


# ============================================================
# 層内相関の測定
# ============================================================
def within_layer_corr(z: torch.Tensor, period: int = 1,
                      pass_len: int = 0) -> torch.Tensor:
    """入力内（位相内）で中心化した z の相関行列を入力平均で返す [D, D]。

    Cov を入力方向に総和してから相関に正規化する（cov_weight の pooling と同じ流儀）。
    """
    cz = phase_center(z, period, pass_len)
    t = z.shape[1]
    cov = torch.einsum("nti,ntj->ij", cz, cz) / t            # [D, D]（入力総和）
    sd = torch.sqrt(torch.diagonal(cov).clamp_min(EPS))
    return cov / (sd.unsqueeze(0) * sd.unsqueeze(1))


def shuffle_time(z: torch.Tensor) -> torch.Tensor:
    """各 (入力, ユニット) について T 方向を独立に並べ替える（帰無分布の生成）。

    同一サンプル内のユニット間結合だけを壊し、周辺分布は保つ。これで測った相関は
    有限サンプルのノイズフロアそのものなので、観測値との差が真の相関の推定になる。
    """
    n, t, d = z.shape
    idx = torch.argsort(torch.rand(n, t, d, device=z.device), dim=1)
    return torch.gather(z, 1, idx)


def corr_summary(R: torch.Tensor) -> dict:
    """相関行列の非対角の大きさと低ランク性（固有値スペクトル）を要約する。"""
    d = R.shape[0]
    off = R - torch.diag(torch.diagonal(R))
    n_off = d * (d - 1)
    ev = torch.linalg.eigvalsh(R.double()).flip(0).clamp_min(0)   # 降順
    total = float(ev.sum())
    # 行和の大きさ: ミラーのバイアスは sum_{k!=i} W_jk rho_ki なので、
    # 個々の |rho| より「符号が揃った和」の方が実害に近い。
    row = off.sum(dim=0)
    return {
        "dim": d,
        "mean_abs_offdiag": float(off.abs().sum() / n_off),
        "max_abs_offdiag": float(off.abs().max()),
        "mean_abs_rowsum": float(row.abs().mean()),
        "ev_top1_frac": float(ev[0] / (total + EPS)),
        "ev_top5_frac": float(ev[:5].sum() / (total + EPS)),
        "ev_top16_frac": float(ev[:16].sum() / (total + EPS)),
        "participation_ratio": float((total ** 2) / ((ev ** 2).sum() + EPS)),
    }


def corr_with_null(z: torch.Tensor, period: int = 1, pass_len: int = 0) -> dict:
    """観測相関とその帰無分布（時間シャッフル）を並べて返す。

    `excess_*` は観測 - 帰無で，有限サンプルのノイズフロアを差し引いた真の相関の目安。
    参加率 PR は帰無で次元 D に一致し，構造があるほど小さくなる。
    """
    obs = corr_summary(within_layer_corr(z, period, pass_len))
    null = corr_summary(within_layer_corr(shuffle_time(z), period, pass_len))
    return {
        "obs": obs, "null": null,
        "excess_mean_abs_offdiag": obs["mean_abs_offdiag"] - null["mean_abs_offdiag"],
        "excess_mean_abs_rowsum": obs["mean_abs_rowsum"] - null["mean_abs_rowsum"],
        "pr_ratio": obs["participation_ratio"] / (null["participation_ratio"] + EPS),
    }


def collect(net, x, passes: int = 1) -> dict:
    """forward フックで per-sample の d, z, y_samples を集める（fncl_mnist_fidelity と同形）。"""
    cap = fncl.Capture(net)
    z = [[] for _ in range(cap.n_hidden)]
    d = [[] for _ in range(cap.n_hidden)]
    ys, y_out = [], []
    with torch.no_grad():
        for _ in range(passes):
            y_out.append(net(x))
            ys.append(cap.y_samples)
            for l in range(cap.n_hidden):
                z[l].append(cap.z[l])
                d[l].append(cap.d[l])
    cap.remove()
    return {"y": torch.stack(y_out, dim=0).mean(dim=0),
            "ys": torch.cat(ys, dim=1),
            "z": [torch.cat(v, dim=1) for v in z],
            "d": [torch.cat(v, dim=1) for v in d]}
