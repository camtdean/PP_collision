# VertexHead: the mathematics

How [`train/downstream/vertexing`](../train/downstream/vertexing) turns the track finder's output into primary and
secondary vertices. Each section says what a step does, gives its equations, notes where it differs from ACTS, and
lists its sources: our code, the matching ACTS v48.2.0 (or sPHENIX) source, and the papers the maths comes from
(numbered, see [References](#references)).

## Overview

```
hits + mask_probs
   │  §2  hard assignment of hits to tracks
   ▼
TrackBuilder (tracks.py)
   │  §3  helix hit fit: Taubin circle, z against arc length, covariance from the hit errors
   │  §4  perigee parameters (d0, z0, φ, θ, q/p) at the beam line
   │  §5  optional: ACTS Kalman refit of the hits + propagation to the perigee (through material)
   │  §6  usable tracks and their weights
   ▼
VertexFinder (vertices.py)
   │  §7–9   helix geometry, Kalman vertex update, adaptive multi-vertex fit
   │  §10    primary vertices: ACTS's adaptive multi-vertex finder (AMVF); PV = first vertex kept
   │  §11    soft cuts
   │  §12    secondary vertices: sPHENIX-style pair seeds, same vertex fit
   │  §13    joint PV + SV refit
   ▼
vertex outputs (model.py: VertexHead.forward)
```

`VertexHead` ([model.py:VertexHead](../train/downstream/vertexing/model.py#L21)) is just `TrackBuilder` followed by
`VertexFinder`; [`from_detector`](../train/downstream/vertexing/model.py#L40) reads the settings from a detector file
(e.g. [`scripts/acts/detectors/sphenix/sphenix.yaml`](../scripts/acts/detectors/sphenix/sphenix.yaml)).

## 1. Conventions and units

| quantity | our code | ACTS |
|---|---|---|
| lengths | cm (the hits' unit) | mm; `MM_PER_CM = 10` at the boundary |
| momentum | GeV | GeV |
| field | T | T |
| track parameters | perigee $(d_0, z_0, \phi, \theta, q/p, t)$ at the beam line $(b_x, b_y)$ | the same bound parameters on a `PerigeeSurface` |

Symbols used throughout:

* $k = 0.299792458\times10^{-2}$ GeV/(T·cm) (`K_CM`), so $p_T = k\,|B_z|\,R$ with $R$ in cm.
* $\mathbf{b} = (b_x, b_y)$: the beam line (`beam_position`, default $(0, 0)$).
* $\rho$: the **signed** helix radius, $\rho = \dfrac{\sin\theta}{(q/p)\,k\,B_z}$, so $\operatorname{sign}\rho = \operatorname{sign}(qB_z)$.
  $\rho > 0$: the track turns clockwise seen from $+z$; $\rho < 0$: anticlockwise. (`R_s` in tracks.py is the same
  quantity.)
* Perigee point: $\mathbf{p}_\perp = \mathbf{b} + d_0(-\sin\phi, \cos\phi)$ (ACTS's sign of $d_0$); circle centre
  $\mathbf{c} = \mathbf{p}_\perp + \rho(\sin\phi, -\cos\phi)$.
* $\operatorname{wrap}(\alpha)$: the angle mapped into $[-\pi, \pi)$.

A check of the signs: a positive track in $+B_z$ moving along $+x$ bends towards $-y$; its centre is
$\mathbf{p}_\perp + \rho(0, -1)$ with $\rho > 0$.

**Sources** — Code: [common.py](../train/downstream/vertexing/common.py#L6) (units, `_wrap`),
[vertices.py:_helix_radius](../train/downstream/vertexing/vertices.py#L51).
ACTS: [Units.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Definitions/Units.hpp),
[PerigeeSurface.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Surfaces/PerigeeSurface.hpp).
Maths: [12].

## 2. Hits and track assignment

The dataset's normalisation is undone in float64, then each real hit is given to its most probable track.

$$
E = E_n\sigma_E + \mu_E,\qquad
\eta, \phi, r \text{ from their min–max ranges},\qquad
x = r\cos\phi,\ y = r\sin\phi,\ z = r\sinh\eta .
$$

A hit goes to track $\arg\max_t$ `mask_probs` if that maximum is $> 0$ (hard assignment). Hits at
$r <$ `silicon_max_radius` (25 cm) are silicon hits, with their own errors.

With a field map, a track's $B_z$ is the mean over its hits of the map's $B_z$ (trilinear interpolation, clamped to the
grid); otherwise the uniform `Bz` (1.4 T).

**Sources** — Code: [tracks.py:_unnormalize_hits](../train/downstream/vertexing/tracks.py#L132),
[tracks.py:TrackBuilder.forward](../train/downstream/vertexing/tracks.py#L529) (assignment, field, silicon),
[tracks.py:_FieldMap.bz](../train/downstream/vertexing/tracks.py#L320).
ACTS: none (input preparation). Maths: —.

## 3. Helix hit fit

Each track's hits are fitted with a helix: a circle in $x$-$y$, then a straight line $z(s)$ in the arc length $s$.
All hits have equal weight; tracks with fewer than `min_hits` (3) hits get NaN.

### 3.1 Circle (Taubin fit, Newton form)

With the hits centred on the track's centroid, $(X_i, Y_i)$, $Z_i = X_i^2 + Y_i^2$, and moments
$M_{ab} = \langle ab\rangle$ over the track's hits:

$$
M_z = M_{XX} + M_{YY},\qquad
\mathrm{Cov}_{XY} = M_{XX}M_{YY} - M_{XY}^2,\qquad
\mathrm{Var}_Z = M_{ZZ} - M_z^2
$$

$$
P(\lambda) = A_3\lambda^3 + A_2\lambda^2 + A_1\lambda + A_0,\qquad
\begin{aligned}
A_3 &= 4M_z, \qquad A_2 = -3M_z^2 - M_{ZZ},\\
A_1 &= \mathrm{Var}_Z M_z + 4\,\mathrm{Cov}_{XY}M_z - M_{XZ}^2 - M_{YZ}^2,\\
A_0 &= M_{XZ}(M_{XZ}M_{YY} - M_{YZ}M_{XY}) + M_{YZ}(M_{YZ}M_{XX} - M_{XZ}M_{XY}) - \mathrm{Var}_Z\,\mathrm{Cov}_{XY}.
\end{aligned}
$$

The smallest root $\lambda^*$ is found by Newton's method from $\lambda = 0$ (at most 20 steps, a step kept only while
$\lvert P\rvert$ decreases). With $D = \lambda^{*2} - \lambda^* M_z + \mathrm{Cov}_{XY}$:

$$
X_c = \frac{M_{XZ}(M_{YY} - \lambda^*) - M_{YZ}M_{XY}}{2D},\qquad
Y_c = \frac{M_{YZ}(M_{XX} - \lambda^*) - M_{XZ}M_{XY}}{2D},\qquad
R = \sqrt{X_c^2 + Y_c^2 + M_z},
$$

and the centre is $\mathbf{c} = (X_c, Y_c) + $ centroid.

### 3.2 Direction of travel and charge

From the innermost hit $\mathbf{h}_\text{in}$ to the outermost $\mathbf{h}_\text{out}$ (by $r$): the track turns
anticlockwise iff $[(\mathbf{h}_\text{in} - \mathbf{c})\times(\mathbf{h}_\text{out} - \mathbf{c})]_z > 0$. This sets
$\operatorname{sign}\rho$ ($-1$ if anticlockwise), $\rho = \pm R$, and $q = \operatorname{sign}(\rho)\operatorname{sign}(B_z)$.

### 3.3 Arc length

Signed arc length from the circle's closest approach to the beam line, positive along the direction of travel:

$$
s = -\rho\,\operatorname{wrap}(\alpha - \alpha_b),\qquad
\alpha = \operatorname{atan2}(y - c_y,\ x - c_x),\qquad
\alpha_b = \operatorname{atan2}(b_y - c_y,\ b_x - c_x).
$$

The wrap limits $\lvert s\rvert \le \pi\lvert\rho\rvert$: a track that turns more than half a circle from its perigee
(a looper, $p_T \lesssim 0.16$ GeV at 1.4 T for the TPC's outer radius of 75 cm) cannot be described.

### 3.4 z line

A helix is linear in arc length, $z = s\cot\theta + z_0$. The line is fitted per track by Deming regression with equal
variances (orthogonal distance). With $S_{ss}, S_{zz}, S_{sz}$ the centred sums of squares and products of $s$ and $z$:

$$
\cot\theta = \frac{S_{zz} - S_{ss} + \sqrt{(S_{zz} - S_{ss})^2 + 4S_{sz}^2}}{2S_{sz}},\qquad
z_0 = \bar z - \bar s\cot\theta .
$$

Then $\theta = \operatorname{atan2}(1, \cot\theta)$, $p_T = k\lvert B_z\rvert R$, $q/p = q\sin\theta/p_T$, and the
circle's distance to the beam line $d_\text{beam} = \big\lvert\,\lvert\mathbf{c} - \mathbf{b}\rvert - R\,\big\rvert$.

### 3.5 Hit-fit covariance

The covariance of $\mathbf{a} = (l_0, l_1, \phi, \theta, q/p)$ at a reference arc length $s_r$, by linearised least
squares on the hit residuals (hit errors only, no material). $l_0$ is along the $d_0$ axis; $l_1$ is $z_0$ (perigee)
or curvilinear loc1 (the ACTS start state, §5.1). Per hit, with

$$
\Delta s = s - s_r,\quad
\kappa = -\frac{kB_z\,(q/p)}{\sin\theta},\quad
a = \kappa\,\Delta s,\quad
k_{q/p} = \frac{kB_z}{\sin\theta},\quad
k_\theta = -\frac{kB_z\,(q/p)\cos\theta}{\sin^2\theta},
$$

$$
\beta = -\tfrac12\Delta s^2\operatorname{sinc}^2\tfrac a2,\qquad
g = \frac{1 - \operatorname{sinc}a}{a}\quad(\operatorname{sinc}x = \sin x / x),
$$

the two residual rows are ($\kappa$: curvature, positive to the left; $a$: turning angle):

| residual | $\partial/\partial l_0$ | $\partial/\partial l_1$ | $\partial/\partial\phi$ | $\partial/\partial\theta$ | $\partial/\partial(q/p)$ |
|---|---|---|---|---|---|
| transverse offset $/\sigma_{r\phi}$ | $\cos a$ | $0$ | $\Delta s\operatorname{sinc}a$ | $\beta k_\theta$ | $\beta k_{q/p}$ |
| $z$ $/\sigma_z$ | $-\cot\theta\sin a$ | $1$ (curvilinear: $1/\sin\theta$) | $-\tfrac12\cot\theta\,\Delta s^2\kappa\operatorname{sinc}^2\tfrac a2$ | $-\dfrac{\Delta s}{\sin^2\theta} + \cot\theta\,\Delta s^2 g\,k_\theta$ | $\cot\theta\,\Delta s^2 g\,k_{q/p}$ |

Information $I = \sum_\text{hits}(\mathbf{r}_\perp\mathbf{r}_\perp^T + \mathbf{r}_z\mathbf{r}_z^T)$, covariance
$C = I^{-1}$. Hit errors: TPC $\sigma_{r\phi} = 0.034$ cm, $\sigma_z = 0.1$ cm; silicon 0.001 cm each.

**Differs from ACTS:** ACTS has no counterpart for this step (it starts tracks from seeds and fits them with its
Kalman filter, §5). z is fitted against the arc length $s$ (exact for a helix), not against $r$.

**Sources** — Code: [tracks.py:_circle_fit](../train/downstream/vertexing/tracks.py#L72),
[tracks.py:_helix_from_hits](../train/downstream/vertexing/tracks.py#L151),
[common.py:_arc_length](../train/downstream/vertexing/common.py#L14),
[tracks.py:_line_fit](../train/downstream/vertexing/tracks.py#L47),
[tracks.py:_track_covariance](../train/downstream/vertexing/tracks.py#L209).
ACTS: none. Maths: Taubin fit [7], its Newton form [8]; Deming regression [9]; least-squares covariance [4].

## 4. Perigee parameters

The fitted circle and line become ACTS perigee parameters at the beam line, and back.

Circle → perigee, with $h = \operatorname{sign}\rho$ and $\mathbf{u} = (\mathbf{b} - \mathbf{c})/\lvert\mathbf{b} - \mathbf{c}\rvert$:

$$
d_0 = h\,(\lvert\rho\rvert - \lvert\mathbf{b} - \mathbf{c}\rvert),\qquad
\phi = \operatorname{atan2}(-h u_x,\ h u_y),\qquad
z_0 \text{ from §3.4},\qquad t = 0 .
$$

Perigee → circle (used on every route; with a field map $B_z$ is taken at the perigee):

$$
p_T = \frac{\sin\theta}{\lvert q/p\rvert},\qquad
\rho = \operatorname{sign}(q/p)\operatorname{sign}(B_z)\frac{p_T}{k\lvert B_z\rvert},\qquad
\mathbf{c} = \mathbf{p}_\perp + \rho(\sin\phi, -\cos\phi),\qquad
d_\text{beam} = \lvert d_0\rvert .
$$

Without ACTS, the track's covariance is the hit-fit covariance (§3.5) at $s_r = 0$ with $l_1 = z_0$.

**Sources** — Code: [tracks.py:_perigee_from_circle](../train/downstream/vertexing/tracks.py#L261),
[tracks.py:_circle_from_perigee](../train/downstream/vertexing/tracks.py#L279).
ACTS: [PerigeeSurface.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Surfaces/PerigeeSurface.hpp).
Maths: [12].

## 5. ACTS track refit and propagation (`use_acts`)

With `use_acts`, ACTS refits each track's hits with its Kalman filter and propagates the result back to the beam line
through the detector material. This runs in a persistent worker process inside the ACTS container (mm throughout).

### 5.1 Start state

At the circle point nearest the innermost hit: position (with $z = s\cot\theta + z_0$), direction
$\hat{\mathbf{d}} = (\sin\theta\,\hat{\mathbf{t}}_\perp,\ \cos\theta)$ ($\hat{\mathbf{t}}_\perp$ the circle's
tangent along travel), the hit fit's $q/p$, and the hit-fit covariance (§3.5) at that $s$ in curvilinear form,
converted to mm ($t$ uncorrelated, variance 1).

### 5.2 Kalman refit (`acts_kalman_refit`, on by default)

* **Prior:** the start state moved 10 mm back along a straight line, covariance $\times 100$
  (`acts_kalman_start_covariance_scale`).
* **One plane per hit**, through the hit, normal $\hat{\mathbf{n}}$ = the hit fit's direction there; axes
  $\hat{\mathbf{u}} = \hat{\mathbf{z}}\times\hat{\mathbf{n}}/\lvert\cdot\rvert$, $\hat{\mathbf{v}} = \hat{\mathbf{n}}\times\hat{\mathbf{u}}$.
* **Measurement** $\mathbf{m} = (0, 0)$ on its plane, with covariance
  $V = P^T\big(\sigma_{r\phi}^2\,\hat{\mathbf{e}}_\phi\hat{\mathbf{e}}_\phi^T + \sigma_z^2\,\hat{\mathbf{z}}\hat{\mathbf{z}}^T\big)P$,
  $P = [\hat{\mathbf{u}}\ \hat{\mathbf{v}}]$, $\hat{\mathbf{e}}_\phi$ the azimuthal unit vector at the hit; each hit
  has its own (TPC or silicon) errors.
* **Material** (`acts_kalman_material`, the TPC gas): each TPC hit's plane carries a slab of gas from halfway to the
  previous TPC hit to halfway to the next (none on silicon hits), giving multiple scattering, and energy loss with
  `acts_energy_loss`.
* **Fit:** `KalmanFitter<EigenStepper, DirectNavigator>` with gain-matrix update and smoother; the smoothed state at the
  innermost measurement is kept.

### 5.3 Propagation to the perigee

Backward from the refitted state (or from §5.1 if the refit failed) to the `PerigeeSurface` at $(b_x, b_y, 0)$. With
a geometry: `Navigator` + `MaterialInteractor` (multiple scattering; energy loss with `acts_energy_loss`); without one,
the field only. The covariance is transported. If the propagation fails, the track is dropped.

### 5.4 Refit fallback

Where the refit failed but the propagation succeeded, the perigee covariance is inflated:
$C \leftarrow s_f\,C$, $s_f = 100$ (`acts_kalman_fallback_covariance_scale`; $\ln s_f$ is learnable with
`learn_weights`).

**Differs from ACTS:** the start state comes from our hit fit (§3), not from ACTS's seeding; the measurement planes are
built per hit (sPHENIX fits on its detector surfaces); the fallback is a soft down-weighting instead of dropping the
track. No gradient flows through ACTS.

**Sources** — Code: [tracks.py:TrackBuilder.forward](../train/downstream/vertexing/tracks.py#L555) (start state, hits,
fallback), [acts.py:_ActsWorker.propagate](../train/downstream/vertexing/acts.py#L61),
[acts_worker.py:build_propagator](../scripts/acts/acts_worker.py#L56),
[perigee_propagator.cpp:propagate](../scripts/acts/perigee_propagator/perigee_propagator.cpp#L150),
[perigee_propagator.cpp:refit](../scripts/acts/perigee_propagator/perigee_propagator.cpp#L279),
[perigee_propagator.cpp:run](../scripts/acts/perigee_propagator/perigee_propagator.cpp#L369).
ACTS: [KalmanFitter.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/TrackFitting/KalmanFitter.hpp),
[GainMatrixUpdater.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/TrackFitting/GainMatrixUpdater.hpp),
[GainMatrixSmoother.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/TrackFitting/GainMatrixSmoother.hpp),
[MaterialInteractor.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Propagator/MaterialInteractor.hpp),
[PerigeeSurface.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Surfaces/PerigeeSurface.hpp).
Maths: Kalman track fit [4], [12].

## 6. Track products

What `TrackBuilder` hands to `VertexFinder`:

* each track's perigee $\mathbf{q} = (d_0, z_0, \phi, \theta, q/p)$ (cm) at the beam line;
* its full $5\times5$ covariance $C$ (cm) $\times$ `track_error_scale`$^2$ (2.5$^2$);
* $B_z$ at the perigee, and its TPC and silicon hit counts.

A track is **usable** if its perigee and covariance are finite, $p_T \ge 0.25$ GeV (`noise_pt_threshold`) and
$\lvert d_0\rvert < 1$ cm (`valid_radius`). Its **weight** is

$$
w = P(\text{real})\cdot\mathbb{1}[\text{usable}],
$$

with `learn_weights`: $\times$ a softplus MLP of (class_probs, $q/(p_T+1)$, $\theta$, $\sin\phi$, $\cos\phi$), and the two
cuts as straight-through gates (hard forward, sigmoid gradient backward). Tracks with $w > 0$ enter the vertex finder;
$w$ multiplies each track's weight in every vertex fit (§9).

**Differs from ACTS:** ACTS's vertexing has no per-track weight ($w = 1$); here it is the track finder's
$P(\text{real})$ and a place for learning.

**Sources** — Code: [tracks.py:TrackBuilder._usable](../train/downstream/vertexing/tracks.py#L498),
[tracks.py:TrackBuilder.forward](../train/downstream/vertexing/tracks.py#L604) (from line 604 on).
ACTS: none. Maths: —.

## 7. Helix geometry for vertexing

Three operations on a track's helix, used by every vertex fit. Their Jacobians $J$ come from automatic
differentiation (`torch.func.jacrev`).

**Perigee at a point** $\mathbf{x}$ (the perigee w.r.t. a line along $z$ through $\mathbf{x}$). With
$\hat{\mathbf{u}} = (\mathbf{x}_\perp - \mathbf{c})/\lvert\mathbf{x}_\perp - \mathbf{c}\rvert$:

$$
\text{PCA}_\perp = \mathbf{c} + \lvert\rho\rvert\hat{\mathbf{u}},\quad
\phi' = \operatorname{atan2}(-h u_x,\ h u_y),\quad
z = z_0 - \rho\operatorname{wrap}(\phi' - \phi)\cot\theta,
$$

$$
d_0' = (\text{PCA}_\perp - \mathbf{x}_\perp)\cdot(-\sin\phi', \cos\phi'),\quad
z_0' = z - x_z,\quad
C' = JCJ^T,\ J = \partial\mathbf{q}'/\partial\mathbf{q}.
$$

**3D impact** at $\mathbf{x}$. On the helix
$\mathbf{h}(\varphi) = \mathbf{O} + \rho(-\sin\varphi, \cos\varphi, -\varphi\cot\theta)$, with
$\mathbf{O} = \big(b_x - (d_0 - \rho)\sin\phi,\ b_y + (d_0 - \rho)\cos\phi,\ z_0 + \rho\phi\cot\theta\big)$,
Newton's method on $\tfrac12\lvert\mathbf{h}(\varphi) - \mathbf{x}\rvert^2$ from $\varphi = \phi$ (20 steps) gives the
closest point $\mathbf{P}$ and direction $\hat{\mathbf{T}} = (\cos\varphi\sin\theta, \sin\varphi\sin\theta, \cos\theta)$.
On the curvilinear axes $\hat{\mathbf{U}} = \hat{\mathbf{z}}\times\hat{\mathbf{T}}/\lvert\cdot\rvert$,
$\hat{\mathbf{V}} = \hat{\mathbf{T}}\times\hat{\mathbf{U}}$:

$$
\mathbf{l} = \big((\mathbf{P} - \mathbf{x})\cdot\hat{\mathbf{U}},\ (\mathbf{P} - \mathbf{x})\cdot\hat{\mathbf{V}}\big),\qquad
C_l = J_l C J_l^T .
$$

For SV seeds (*guarded*), Newton also starts from the transverse closest approach on the three turns nearest the point
in $z$, and the closest true minimum wins (far points, low $p_T$).

**Linearisation** at $\mathbf{x}_0$ (Billoir–Qian perigee parametrisation, Piacquadio's helix Jacobians). With the
perigee there $\mathbf{q}'$, its PCA $\mathbf{p}$ (3D), $X = p_x - x_{0,x} + \rho\sin\phi'$,
$Y = p_y - x_{0,y} - \rho\cos\phi'$, $S^2 = X^2 + Y^2$, $h = \operatorname{sign}\rho$, the perigee parameters as a
function of vertex $\mathbf{x}$ and momentum $\boldsymbol{\pi} = (\phi, \theta, q/p)$ are

$$
\mathbf{q}(\mathbf{x}, \boldsymbol{\pi}) \approx \mathbf{c}_0 + A\mathbf{x} + B\boldsymbol{\pi},\qquad
A = \begin{pmatrix} -hX/S & -hY/S & 0\\ \rho\cot\theta\,Y/S^2 & -\rho\cot\theta\,X/S^2 & 1\\ -Y/S^2 & X/S^2 & 0\\ 0&0&0\\ 0&0&0 \end{pmatrix},\quad
B = \begin{pmatrix} 0&0&0\\ \rho\cot\theta\,(1 - \lvert\rho\rvert/S) & 0 & 0\\ \lvert\rho\rvert/S & 0 & 0\\ 0&1&0\\ 0&0&1 \end{pmatrix},
$$

$\mathbf{c}_0 = \mathbf{q}' - A\mathbf{p} - B\boldsymbol{\pi}'$, and the track weight matrix $G = C'^{-1}$.

**Differs from ACTS:** ACTS propagates numerically to the perigee surface; here the helix is transported
analytically (uniform $B_z$ per track, the value at its perigee), in cm.

**Sources** — Code: [vertices.py:_perigee_at](../train/downstream/vertexing/vertices.py#L56),
[vertices.py:_impact_3d](../train/downstream/vertexing/vertices.py#L72),
[vertices.py:_transport](../train/downstream/vertexing/vertices.py#L137),
[vertices.py:_linearise](../train/downstream/vertexing/vertices.py#L144).
ACTS: [HelicalTrackLinearizer.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/HelicalTrackLinearizer.cpp),
[ImpactPointEstimator.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/ImpactPointEstimator.cpp).
Maths: [2], [6].

## 8. Kalman vertex update

The vertex position from a prior and the linearised tracks, each with a weight $\omega_i$ (fit weight $\times$ track
weight $w_i$). With

$$
W_i = (B_i^T G_i B_i)^{-1},\qquad G_i^B = G_i - G_i B_i W_i B_i^T G_i,
$$

the prior $(\mathbf{x}_p, \sigma_p^2 I)$ ($\sigma_p^2$ = `initial_variance` = 1 cm², ACTS's $10^2$ mm²):

$$
C_v^{-1} = \sigma_p^{-2}I + \sum_i\omega_i A_i^T G_i^B A_i,\qquad
\hat{\mathbf{x}} = C_v\Big(\sigma_p^{-2}\mathbf{x}_p + \sum_i\omega_i A_i^T G_i^B(\mathbf{q}_i - \mathbf{c}_{0,i})\Big).
$$

This is the one-track-at-a-time Kalman update written as one sum (same result). Residual
$\mathbf{r}_i = \mathbf{q}_i - \mathbf{c}_{0,i} - A_i\hat{\mathbf{x}}$, and

$$
\chi^2 = \frac{\lvert\hat{\mathbf{x}} - \mathbf{x}_p\rvert^2}{\sigma_p^2} + \sum_i\omega_i\,\mathbf{r}_i^T G_i^B\mathbf{r}_i,\qquad
\text{ndf} = 2\sum_i\omega_i - 3 .
$$

Each track's momentum refitted to the vertex: $\tilde{\boldsymbol{\pi}}_i = W_i B_i^T G_i\,\mathbf{r}_i$ (used for SV
momenta, §12).

**Sources** — Code: [vertices.py:VertexFinder._kalman](../train/downstream/vertexing/vertices.py#L393),
[vertices.py:VertexFinder._vertex_momentum](../train/downstream/vertexing/vertices.py#L704).
ACTS: [KalmanVertexUpdater.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/KalmanVertexUpdater.cpp),
[KalmanVertexUpdater.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Vertexing/KalmanVertexUpdater.hpp).
Maths: [4], [5], [6].

## 9. Adaptive multi-vertex fit

Several vertices are fitted together; each track's weight is shared between the vertices it may belong to, so a
track pulled by two vertices counts mostly for the closer one.

Each vertex $v$ keeps its tracks' 3D impact $\mathbf{l}_i$ and weight $C_l^{-1}$ at its seed $\mathbf{s}_v$ (§7). In
each iteration, with temperature $T$:

1. Relinearise a vertex's tracks if it moved more than 0.05 cm in $x$-$y$ from its linearisation point
   (`max_dist_to_lin_point`).
2. Compatibility of track $i$ with vertex $v$ at its current position $\mathbf{x}_v$:
   $$
   \chi^2_{iv} = \mathbf{e}^T C_l^{-1}\mathbf{e},\qquad
   \mathbf{e} = \mathbf{l}_i - \big((\mathbf{x}_v - \mathbf{s}_v)\cdot\hat{\mathbf{U}},\ (\mathbf{x}_v - \mathbf{s}_v)\cdot\hat{\mathbf{V}}\big).
   $$
3. Annealed weight, over the vertices the track is linked to ($f_{iv} \in [0, 1]$: membership, §10), cut-off
   $\chi^2_c = 9$:
   $$
   \omega_{iv} = \frac{f_{iv}\,e^{-\chi^2_{iv}/2T}}{e^{-\chi^2_c/2T} + \sum_{v'}f_{iv'}\,e^{-\chi^2_{iv'}/2T}} ;
   $$
   $\omega \le 0.001$ (`fit_min_weight`) is set to 0.
4. Kalman update (§8) of every vertex with weights $\omega_{iv}w_i$.

Temperatures: one, $T = 1$ (as ACTS's example algorithm). The fit stops after the last temperature once every vertex's
shift satisfies $\Delta\mathbf{x}^T C_v^{-1}\Delta\mathbf{x} \le 0.01$ (`max_relative_shift`), or after 30 iterations.
With `shared=False` (SV seeds) the sum in the denominator has only the vertex's own term.

**Sources** — Code: [vertices.py:VertexFinder._fit](../train/downstream/vertexing/vertices.py#L415),
[vertices.py:VertexFinder._compatibility](../train/downstream/vertexing/vertices.py#L375),
[vertices.py:VertexFinder._relinearise](../train/downstream/vertexing/vertices.py#L382),
[vertices.py:VertexFinder._new_vertices](../train/downstream/vertexing/vertices.py#L338).
ACTS: [AdaptiveMultiVertexFitter.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/AdaptiveMultiVertexFitter.cpp),
[AdaptiveMultiVertexFitter.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Vertexing/AdaptiveMultiVertexFitter.hpp),
[AnnealingUtility.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Utilities/AnnealingUtility.cpp),
[ImpactPointEstimator.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/ImpactPointEstimator.cpp)
(3D compatibility).
Maths: [1], [3].

## 10. Primary vertices: adaptive multi-vertex finder

Vertices are found one at a time: seed a $z$ from the track density, attach compatible tracks, fit, keep or delete,
and repeat on the tracks not yet used. The settings default to ACTS's `AdaptiveMultiVertexFinderAlgorithm` (Gaussian
seeder), converted to cm.

**1. Seed** (Gaussian track density, over the unused tracks). Each track contributes a Gaussian in $z$ from its
$(d_0, z_0)$ covariance $\Sigma = \begin{pmatrix}c_{dd} & c_{dz}\\ c_{dz} & c_{zz}\end{pmatrix}$, $\det\Sigma = \Delta$:

$$
\ln\rho_i(z) = c_0 + c_1 z + c_2 z^2 + \ln\frac{w_i}{2\pi\sqrt\Delta},\qquad
c_0 = -\frac{d_0^2c_{zz} + z_0^2c_{dd} + 2d_0z_0c_{dz}}{2\Delta},\quad
c_1 = \frac{d_0c_{dz} + z_0c_{dd}}{\Delta},\quad
c_2 = -\frac{c_{dd}}{2\Delta},
$$

used only if $d_0^2/c_{dd} \le 3.5^2$ and only inside its window $c_0 + c_1z + c_2z^2 \ge -2\cdot12^2$. The density
$\sum_i\rho_i(z)$ is evaluated at each track's $z_0$ and after each of two Newton steps on its logarithm,
$z \leftarrow z + \rho\rho'/(\rho'^2 - \rho\rho'')$; the highest value wins (first in track order on ties). The seed
is $(b_x, b_y, z)$, which is also the vertex prior.

**2. Preselection** (membership $f_i$, soft, over all input tracks):

$$
f_i = S_\text{below}\big(\lvert z_{0,i} - z_\text{seed}\rvert;\ 0.1\ \text{cm}\big)\cdot
S_\text{below}\Big(\sqrt{(d_0'/\sigma_{d_0'})^2 + (z_0'/\sigma_{z_0'})^2};\ 5\Big),
$$

with $d_0', z_0'$ and their errors from the perigee at the seed (§7). A track is linked if $f_i > 0.5$. If no track is
linked, the seed moves to the nearest unused track's $z_0$ (the prior stays); if still none, the search stops.

**3. Fit** the new vertex together with every kept vertex it shares linked tracks with (§9).

**4. Keep or delete.** The new vertex's linked, unused tracks with $\chi^2 < 18.42$ (`max_vertex_chi2`) leave the pool
(if there are none: its least compatible unused track, or else the unused track nearest in $z$). The vertex is kept if

* at least 2 such tracks,
* contamination $\sum_i\omega_i(1 - \omega_i)/\sum_i\omega_i^2 \le 0.5$,
* $\lvert z - z_k\rvert \ge 3\sqrt{\sigma_z^2 + \sigma_{z,k}^2}$ for every kept vertex $k$ (not merged).

Otherwise it is deleted and the vertices it was fitted with are refitted without it.

**5. Repeat** while at least 2 tracks are unused and fewer than `max_vertices` (10) are kept.

The **PV** is the first vertex kept. All kept vertices are output (`primary_vertices`) with each track's vertex.

**PV fallback.** Events where no vertex is kept are searched again with a looser preselection, step by step
(`fallback_preselection`: (z interval, significance) = (0.3 cm, 7), then (1 cm, 10)); vertices are still kept by the
rules of step 4. `vertex_fallback_step` records the step (0: as ACTS, −1: none). Off with `hard_cuts` or
`pv_fallback=False`.

**Differs from ACTS:** soft membership cuts (§11); a per-track weight $w$ (§6) in the density and the fit; the PV
fallback with relaxed preselection (ACTS returns no vertex); a cap of `max_vertices`; lengths in cm. There is no
beam-spot constraint, as in ACTS's shipped AMVF: the prior is the seed with 1 cm² in each coordinate.

**Sources** — Code: [vertices.py:_density_seed](../train/downstream/vertexing/vertices.py#L173),
[vertices.py:VertexFinder._links](../train/downstream/vertexing/vertices.py#L458),
[vertices.py:VertexFinder._find_primary](../train/downstream/vertexing/vertices.py#L469),
[vertices.py:VertexFinder.forward](../train/downstream/vertexing/vertices.py#L737) (fallback, PV).
ACTS: [AdaptiveMultiVertexFinder.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/AdaptiveMultiVertexFinder.cpp)
(`doSeeding`, `canRecoverFromNoCompatibleTracks`, `keepNewVertex`, `isMergedVertex`),
[AdaptiveMultiVertexFinder.hpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/include/Acts/Vertexing/AdaptiveMultiVertexFinder.hpp),
[GaussianTrackDensity.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/GaussianTrackDensity.cpp),
[TrackDensityVertexFinder.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/TrackDensityVertexFinder.cpp),
[AdaptiveMultiVertexFinderAlgorithm.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Examples/Algorithms/Vertexing/src/AdaptiveMultiVertexFinderAlgorithm.cpp)
and its [header](https://github.com/acts-project/acts/blob/v48.2.0/Examples/Algorithms/Vertexing/include/ActsExamples/Vertexing/AdaptiveMultiVertexFinderAlgorithm.hpp)
(the shipped settings).
Maths: [3], [10], [1].

## 11. Soft cuts

Every cut that decides track membership or an SV seed is a sigmoid, so gradients can flow through it:

$$
S_\text{below}(v; c) = \sigma\!\left(\frac{c - v}{\epsilon}\right),\qquad
S_\text{above}(v; c) = \sigma\!\left(\frac{v - c}{\epsilon}\right),
$$

with width $\epsilon$ = `cut_softness_cm` (0.005 cm), `cut_softness_sigma` (0.25 σ) or `cut_softness_hits` (1 hit).
A cut on a $\chi^2$ uses $\epsilon = 2\sqrt{\chi^2_c}\cdot0.25$ (0.25 σ expressed in $\chi^2$). NaN gives 0. With
`hard_cuts=True` the sigmoids become steps, which reproduces ACTS. The keep/delete decisions of §10 step 4 and the SV
selection of §12 step 4 stay hard.

**Differs from ACTS:** ACTS uses hard cuts throughout.

**Sources** — Code: [vertices.py:_soft_below](../train/downstream/vertexing/vertices.py#L29),
[vertices.py:VertexFinder._width](../train/downstream/vertexing/vertices.py#L329).
ACTS: none (ACTS cuts are hard). Maths: —.

## 12. Secondary vertices

Displaced tracks are paired where their circles cross; each pair seeds a vertex fitted with the same machinery (§9).
The selection follows sPHENIX's `SecondaryVertexFinder` (written from its description, not its code).

**1. Candidates** (soft), from the tracks not used by the other primary vertices (their weight there $< 0.5$):

* at least `sv_min_track_hits` (20) TPC hits;
* displaced from the PV: with silicon hits, $\lvert d_0'\rvert > 0.02$ cm and $\lvert z_0'\rvert > 0.02$ cm at the PV
  (sPHENIX's cuts); TPC-only tracks (too coarse for those), compatibility with the PV $\chi^2_\text{PV} > 9$.

**2. Pair seeds.** For candidate circles $i, j$ (centres $\mathbf{c}_{1,2}$, radii $r_{1,2} = \lvert\rho\rvert$,
$d = \lvert\mathbf{c}_2 - \mathbf{c}_1\rvert$, $\hat{\mathbf{e}} = (\mathbf{c}_2 - \mathbf{c}_1)/d$), both intersections

$$
\mathbf{x}_\perp = \mathbf{c}_1 + a\hat{\mathbf{e}} \pm h\hat{\mathbf{e}}_\perp,\qquad
a = \frac{r_1^2 - r_2^2 + d^2}{2d},\quad h = \sqrt{r_1^2 - a^2}.
$$

At each, both helices give a point $\mathbf{x}_{1,2}$ (with $z = z_0 - \rho\operatorname{wrap}(\phi_\text{at} - \phi)\cot\theta$)
and tangent $\hat{\mathbf{u}}_{1,2}$. The tangent lines' closest approach, with
$\mathbf{w} = \mathbf{x}_1 - \mathbf{x}_2$, $b = \hat{\mathbf{u}}_1\cdot\hat{\mathbf{u}}_2$,
$d_k = \hat{\mathbf{u}}_k\cdot\mathbf{w}$:

$$
t_1 = \frac{b d_2 - d_1}{1 - b^2},\quad t_2 = \frac{d_2 - b d_1}{1 - b^2},\qquad
\text{DCA} = \lvert\mathbf{x}_1 + t_1\hat{\mathbf{u}}_1 - \mathbf{x}_2 - t_2\hat{\mathbf{u}}_2\rvert,
$$

and the seed is the midpoint of those two points. A seed passes if $\lvert\mathbf{x}_\perp\rvert < 40$ cm,
$\lvert z_1 - z_2\rvert < 1$ cm, DCA $< 0.5$ cm and its distance from the PV $> 0.2$ cm (all soft). The 6
(`max_sv_per_event`) passing seeds with the smallest DCA are kept.

**3. Fit.** Each seed is fitted on its own (§9 with `shared=False`, guarded 3D impact) with its pair and the
candidates of $\chi^2 \le 9$ at the seed (soft).

**4. Selection.** Kept if at least 2 linked tracks have $\chi^2 < 18.42$, ndf $> 0$, $\chi^2/\text{ndf} \le 10$ and
decay length $> 0.2$ cm. Of SVs closer than 0.1 cm to each other only the lowest $\chi^2/\text{ndf}$ is kept.

**5. Outputs** (KFParticle-style topology). With $\mathbf{L} = \mathbf{x}_\text{SV} - \mathbf{x}_\text{PV}$,
$\hat{\mathbf{L}} = \mathbf{L}/L$, and the tracks of weight $\ge 0.5$:

$$
\mathbf{p} = \sum_i\frac{(\cos\tilde\phi_i\sin\tilde\theta_i,\ \sin\tilde\phi_i\sin\tilde\theta_i,\ \cos\tilde\theta_i)}{\lvert\widetilde{q/p}_i\rvert},\qquad
\text{DIRA} = \frac{\hat{\mathbf{L}}\cdot\mathbf{p}}{\lvert\mathbf{p}\rvert},\qquad
\frac{L}{\sigma_L} = \frac{L}{\sqrt{\hat{\mathbf{L}}^T(C_\text{SV} + C_\text{PV})\hat{\mathbf{L}}}},
$$

with the refitted momenta $\tilde{\boldsymbol{\pi}}_i$ of §8; plus the seed's pair DCA.

**Differs from sPHENIX/ACTS:** soft candidate and seed cuts; the TPC-only displacement test by $\chi^2_\text{PV}$; helix
(not straight-line) intersections, tangents and 3D impact; the ACTS vertex fit (§9) for every seed.

**Sources** — Code: [vertices.py:VertexFinder._find_secondary](../train/downstream/vertexing/vertices.py#L617),
[vertices.py:VertexFinder._sv_seeds](../train/downstream/vertexing/vertices.py#L559),
[vertices.py:VertexFinder._sv_outputs](../train/downstream/vertexing/vertices.py#L681),
[vertices.py:VertexFinder._vertex_momentum](../train/downstream/vertexing/vertices.py#L704).
sPHENIX: [SecondaryVertexFinder.cc](https://github.com/sPHENIX-Collaboration/coresoftware/blob/master/offline/packages/trackreco/SecondaryVertexFinder.cc),
[SecondaryVertexFinder.h](https://github.com/sPHENIX-Collaboration/coresoftware/blob/master/offline/packages/trackreco/SecondaryVertexFinder.h)
(master branch; not pinned). ACTS: as §9.
Maths: decay-length significance and DIRA as in KFParticle [11]; vertex fit [1], [4].

## 13. Joint PV + SV refit

In events with a PV and at least one SV, the PV and its SVs are fitted together (§9 with shared weights, each vertex
keeping its own linked tracks). A track linked to both is shared by weight, which removes the SV's tracks' pull on
the PV. The PV and the SV outputs (§12 step 5, measured from the refitted PV) are then taken from this refit; the SV
selection is unchanged. Off with `refit_pv=False`.

**Differs from ACTS / sPHENIX:** neither refits the PV with the SVs; this step is ours.

**Sources** — Code: [vertices.py:VertexFinder.forward](../train/downstream/vertexing/vertices.py#L765).
ACTS: [AdaptiveMultiVertexFitter.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Core/src/Vertexing/AdaptiveMultiVertexFitter.cpp)
(the multi-vertex fit reused). Maths: [1].

## 14. ACTS reference run

The check against ACTS runs ACTS's own `AdaptiveMultiVertexFinderAlgorithm` (Gaussian seeder, as shipped) on the same
perigee tracks (mm) and compares its first vertex with VertexHead's PV; `hard_cuts=True` is the like-for-like setting.

**Sources** — Code: [acts_vertexing.py:main](../scripts/acts/acts_vertexing.py#L22),
[compare_vertexing.py:main](../scripts/acts/compare_vertexing.py#L277).
ACTS: [AdaptiveMultiVertexFinderAlgorithm.cpp](https://github.com/acts-project/acts/blob/v48.2.0/Examples/Algorithms/Vertexing/src/AdaptiveMultiVertexFinderAlgorithm.cpp).
Maths: [12].

## References

Licences: ACTS is MPL-2.0, KFParticle GPL-3.0, sPHENIX's coresoftware has no licence. This code is written from the
published maths below and the algorithms' descriptions, and checked against ACTS's output.

1. W. Waltenberger, R. Frühwirth, P. Vanlaer, *Adaptive vertex fitting*, J. Phys. G 34 (2007) N343.
   [doi:10.1088/0954-3899/34/12/N01](https://doi.org/10.1088/0954-3899/34/12/N01)
2. G. Piacquadio, *Identification of b-jets and investigation of the discovery potential of a Higgs boson in the
   WH → lνbb̄ channel with the ATLAS experiment*, PhD thesis, Freiburg (2010), CERN-THESIS-2010-027.
   [cds.cern.ch/record/1243771](https://cds.cern.ch/record/1243771)
3. G. Piacquadio, K. Prokofiev, A. Wildauer, *Primary vertex reconstruction in the ATLAS experiment at LHC*,
   J. Phys. Conf. Ser. 119 (2008) 032033. [doi:10.1088/1742-6596/119/3/032033](https://doi.org/10.1088/1742-6596/119/3/032033)
4. R. Frühwirth, *Application of Kalman filtering to track and vertex fitting*, Nucl. Instrum. Meth. A 262 (1987) 444.
   [doi:10.1016/0168-9002(87)90887-4](https://doi.org/10.1016/0168-9002(87)90887-4)
5. P. Billoir, R. Frühwirth, M. Regler, *Track element merging strategy and vertex fitting in complex modular
   detectors*, Nucl. Instrum. Meth. A 241 (1985) 115. [doi:10.1016/0168-9002(85)90523-6](https://doi.org/10.1016/0168-9002(85)90523-6)
6. P. Billoir, S. Qian, *Fast vertex fitting with a local parametrization of tracks*, Nucl. Instrum. Meth. A 311
   (1992) 139. [doi:10.1016/0168-9002(92)90859-3](https://doi.org/10.1016/0168-9002(92)90859-3)
7. G. Taubin, *Estimation of planar curves, surfaces, and nonplanar space curves defined by implicit equations, with
   applications to edge and range image segmentation*, IEEE Trans. PAMI 13 (1991) 1115.
   [doi:10.1109/34.103273](https://doi.org/10.1109/34.103273)
8. N. Chernov, *Circular and Linear Regression: Fitting Circles and Lines by Least Squares*, CRC Press (2010).
   [doi:10.1201/EBK1439835906](https://doi.org/10.1201/EBK1439835906)
9. W. E. Deming, *Statistical Adjustment of Data*, Wiley (1943).
10. ATLAS Collaboration, *Development of ATLAS Primary Vertex Reconstruction for LHC Run 3*, ATL-PHYS-PUB-2019-015
    (Gaussian track-density seeder and AMVF). [cds.cern.ch/record/2670380](https://cds.cern.ch/record/2670380)
11. S. Gorbunov, I. Kisel, *Reconstruction of decayed particles based on the Kalman filter*, CBM-SOFT-note-2007-003
    (2007); KFParticle source: [github.com/cbmsw/KFParticle](https://github.com/cbmsw/KFParticle)
12. X. Ai et al., *A Common Tracking Software Project*, Comput. Softw. Big Sci. 6 (2022) 8.
    [doi:10.1007/s41781-021-00078-8](https://doi.org/10.1007/s41781-021-00078-8),
    [arXiv:2106.13593](https://arxiv.org/abs/2106.13593)
