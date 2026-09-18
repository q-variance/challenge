"""Semi-analytic q-variance for additive (independent-increment) processes.

Key identity: for independent increments r_1..r_N with cumulants Psi_i(p) and X = sum r_i,

    E[ sum_i r_i^2 e^{ipX} ] = phi_X(p) sum_i ( -Psi_i''(p) - Psi_i'(p)^2 ),

so E[sum r_i^2 | X = x] f_X(x) is one inverse Fourier transform away. The challenge's
variance estimate is 252/N * sum (r_i - rbar)^2 = 252/N * (sum r_i^2 - X^2/N).
In the limit N -> inf this becomes E[QV | X] f = F^{-1}[-Psi'' phi].

expected_score() evaluates the infinite-sample limit of qvar.score() for anchored
windows (each window starts at the process origin), pooling horizons with weight 1/T
as a single series cut into non-overlapping windows would.
"""
import numpy as np
from .qvar import HORIZONS, S0, Z0, qvar

DAYS = 252


class MeixnerSato:
    """Meixner(alpha = 2 A sqrt(t), beta, delta) Sato process, centred.

    delta = 1/2 gives exact q-variance with coefficient 1/2 under continuous monitoring;
    in general the coefficient is 1 / (1 + 2 delta).
    """

    def __init__(self, A, beta, delta=0.5):
        self.A, self.beta, self.delta = A, beta, delta

    @classmethod
    def from_qvar(cls, s0=S0, z0=Z0):
        return cls(s0 * np.sqrt(2.0), -2.0 * np.arctan(z0 / (s0 * np.sqrt(2.0))), 0.5)

    def cumulant(self, p, t):
        A, b, d = self.A, self.beta, self.delta
        a = A * np.sqrt(t)
        w = a * p - 0.5j * b
        mu = -2 * d * a * np.tan(b / 2)
        ws = np.where(np.real(w) < 0, -w, w)          # cosh is even; use Re(ws) >= 0
        e = np.exp(-2 * ws)
        logcosh = ws + np.log1p(e) - np.log(2.0)
        psi = 2 * d * (np.log(np.cos(b / 2)) - logcosh) + 1j * mu * p
        d1 = -2 * d * a * np.tanh(w) + 1j * mu
        d2 = -2 * d * (a * 2 * np.exp(-ws) / (1 + e)) ** 2
        return psi, d1, d2

    def increment(self, p, ta, tb):
        c0 = self.cumulant(p, ta) if ta > 0 else (0, 0, 0)
        c1 = self.cumulant(p, tb)
        return tuple(y - x for x, y in zip(c0, c1))


class FRHSato:
    """FRH / NIG additive process with gamma(t) = c t^H, log-price martingale version.

    Generator (per unit time) with g = gamma(u), S = sqrt(1 + g^2 s^2 (1-rho^2) p^2 - 2 i g rho s p):
        Lambda = (1 - S)/g^2 - i p rho s / g,
    written in a form that is stable as g -> 0 (where it tends to -s^2 p^2 / 2).
    convexity=True adds the -1/2 s^2 v term of the risk-neutral model (exp(X) martingale).
    """

    def __init__(self, sigma, rho, c, H=0.5, convexity=False, nodes=24):
        self.s, self.rho, self.c, self.H, self.conv = sigma, rho, c, H, convexity
        self.x, self.w = np.polynomial.legendre.leggauss(nodes)

    def _gen(self, p, g):
        """Lambda(p), Lambda'(p), Lambda''(p) at gamma = g (stable as g -> 0).

        With k = s^2/2 (convexity) or 0, E = s^2(1-rho^2)p^2 - 2ip(rho s/g - k),
        S = sqrt(1 + g^2 E), Q = (1 - S)/g = -(g s^2(1-rho^2)p^2 - 2ip(rho s - k g))/(1+S):
            Lambda   = -(s^2(1-rho^2)p^2 + 2ipk)/(1+S) + i p rho s Q/(1+S)
            Lambda'  = -(s^2(1-rho^2)p + ik)/S + i rho s Q/S
            Lambda'' = -s^2(1-rho^2)/S + (g s^2(1-rho^2)p - i rho s + i g k)^2/S^3
        """
        s, r = self.s, self.rho
        k = 0.5 * s ** 2 if self.conv else 0.0
        v = s ** 2 * (1 - r ** 2)
        S = np.sqrt(1 + g ** 2 * v * p ** 2 - 2j * p * (r * s * g - k * g ** 2))
        Q = -(g * v * p ** 2 - 2j * p * (r * s - k * g)) / (1 + S)
        lam = -(v * p ** 2 + 2j * p * k) / (1 + S) + 1j * p * r * s * Q / (1 + S)
        d1 = -(v * p + 1j * k) / S + 1j * r * s * Q / S
        d2 = -v / S + (g * v * p - 1j * r * s + 1j * g * k) ** 2 / S ** 3
        return lam, d1, d2

    def increment(self, p, ta, tb):
        # integrate over u in [ta, tb] with Gauss-Legendre in w = u^(1/2)
        wa, wb = np.sqrt(ta), np.sqrt(tb)
        out = [0, 0, 0]
        for wi, xi in zip(self.w, self.x):
            w = 0.5 * (wb - wa) * xi + 0.5 * (wb + wa)
            u = w * w
            g = self.c * u ** self.H
            vals = self._gen(p, np.maximum(g, 1e-12))
            jac = 0.5 * (wb - wa) * wi * 2 * w
            for k in range(3):
                out[k] = out[k] + jac * vals[k]
        return tuple(out)


def conditional(model, T, n_steps=None, zlim=8.0, n=2 ** 14):
    """z grid, density f(z) and m(z) = E[annualised variance | z] for an anchored window.

    n_steps=None: continuous monitoring (quadratic variation / T); otherwise the
    challenge estimator with n_steps equal sub-periods (n_steps=T for daily data).
    z = X / sqrt(T/252) is not re-centred (models here have E X = 0 unless convexity).
    """
    tT = T / DAYS
    dz = 2 * zlim / n
    z = (np.arange(n) - n // 2) * dz
    x = z * np.sqrt(tT)
    dx = dz * np.sqrt(tT)
    p = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    steps = [(0.0, tT)] if n_steps is None else \
        [(tT * i / n_steps, tT * (i + 1) / n_steps) for i in range(n_steps)]
    psi = 0
    acc = 0
    for ta, tb in steps:
        c = model.increment(p, ta, tb)
        psi = psi + c[0]
        acc = acc + (-c[2] - (0 if n_steps is None else c[1] ** 2))
    phi = np.exp(psi)
    shift = np.exp(-1j * p * x[0])  # f(x0 + k dx) = fft(phi e^{-ip x0})[k] / (n dx)

    def inv(h):
        return np.real(np.fft.fft(h * shift)) / (n * dx)

    fx = inv(phi)
    num = inv(acc * phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = num / fx
    if n_steps is not None:
        cond = cond - x ** 2 / n_steps
    m = cond / tT
    f = fx * np.sqrt(tT)
    return z, f, m


def expected_score(model, horizons=HORIZONS, zmax=0.6, delz=0.05, recentre=True, **kw):
    """Infinite-sample version of qvar.score for anchored windows of the model."""
    bins = np.linspace(-zmax, zmax, int(2 * zmax / delz + 1))
    nb = len(bins) - 1
    W = np.zeros(nb); Wz = np.zeros(nb); Wv = np.zeros(nb)
    bs = []
    for T in horizons:
        z, f, m = conditional(model, T, n_steps=T, **kw)
        ok = np.isfinite(m) & (f > 1e-14)
        z, f, m = z[ok], f[ok], m[ok]
        dz = z[1] - z[0]
        mass = f.sum() * dz
        f = f / mass
        if recentre:
            z = z - np.sum(z * f) * dz
        wT = 1.0 / T
        idx = np.digitize(z, bins) - 1
        sel = (idx >= 0) & (idx < nb)
        np.add.at(W, idx[sel], wT * f[sel] * dz)
        np.add.at(Wz, idx[sel], wT * (z * f)[sel] * dz)
        np.add.at(Wv, idx[sel], wT * (m * f)[sel] * dz)
        Ez2 = np.sum(z ** 2 * f) * dz
        Ez4 = np.sum(z ** 4 * f) * dz
        Em = np.sum(m * f) * dz
        Emz2 = np.sum(m * z ** 2 * f) * dz
        bs.append((Emz2 - Em * Ez2) / (Ez4 - Ez2 ** 2))
    zm, vm = Wz / W, Wv / W
    fit = qvar(zm)
    r2 = 1 - np.sum((vm - fit) ** 2) / np.sum((vm - vm.mean()) ** 2)
    bs = np.array(bs)
    return dict(r2=r2, mae_b=float(np.mean(np.abs(bs - 0.5))), b=bs, z=zm, var=vm)


class IncrementSampler:
    """Inverse-CDF sampler for the daily increments of an anchored additive model.

    Day i (i = 1..n_days) covers [(i-1)/252, i/252] years from the process origin; its
    density is obtained from model.increment() by FFT and tabulated once, so any
    additive model with a characteristic function can be simulated exactly up to
    the grid resolution. Increments are centred (log-price martingale).
    """

    def __init__(self, model, n_days, n=2 ** 17, xlim=16.0):
        self.tables = []
        for i in range(n_days):
            ta, tb = i / DAYS, (i + 1) / DAYS
            sd = np.sqrt(tb) * 0.6  # late-day increments are rare large jumps of size ~ sqrt(t)
            dx = 2 * xlim * sd / n
            x = (np.arange(n) - n // 2) * dx
            p = 2 * np.pi * np.fft.fftfreq(n, d=dx)
            phi = np.exp(model.increment(p, ta, tb)[0])
            f = np.real(np.fft.fft(phi * np.exp(-1j * p * x[0]))) / (n * dx)
            f = np.maximum(f, 0.0)
            F = np.cumsum(f) * dx
            F /= F[-1]
            mean = np.sum(x * f) * dx / (np.sum(f) * dx)
            keep = np.concatenate([[True], np.diff(F) > 0])
            self.tables.append((F[keep], x[keep] + dx / 2 - mean))

    def sample(self, n_paths, n_days, rng=None):
        rng = np.random.default_rng(rng)
        out = np.empty((n_paths, n_days))
        for i in range(n_days):
            F, x = self.tables[i]
            out[:, i] = np.interp(rng.random(n_paths), F, x)
        return out
