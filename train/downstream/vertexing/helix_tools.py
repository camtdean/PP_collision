"""
helix_tools.py: units, the helix geometry and the soft cuts the parts share (cm, the hits' unit).
"""
import functools
import math

import torch
from torch.func import jacrev, vmap

# units: positions in cm (the hits are in cm); ACTS lengths in mm; pT [GeV] = C_LIGHT x B [T] x R [m]
CM_PER_M, MM_PER_CM, C_LIGHT = 100.0, 10.0, 0.299792458
K_CM = C_LIGHT / CM_PER_M          # pT [GeV] = K_CM x B [T] x R [cm]
_CURVILINEAR_TOLERANCE = 0.999995  # |direction . z| above which the curvilinear frame uses x instead of z

def _wrap(angle):
    """Angle in [-pi, pi)."""
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi

def _arc_length(x, y, xc, yc, R_s, beam_position):
    """
    Signed arc length along a circle (centre xc, yc; signed radius R_s, sign(R_s) = sign(q Bz)) from its
    closest approach to the beam line (at beam_position in x-y, cm) to the circle point at the angle of
    (x, y); positive in the direction of travel (anticlockwise <=> R_s < 0). All inputs broadcast.
    """
    beam_angle = torch.atan2(-(yc - beam_position[1]), -(xc - beam_position[0]))
    return _wrap(torch.atan2(y - yc, x - xc) - beam_angle) * -R_s

def _circle_point(circle_centre, R_s, xy):
    """Each circle's point nearest xy (..., 2) and the unit direction of travel there (anticlockwise <=> R_s < 0)."""
    n = xy - circle_centre
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return circle_centre + R_s.abs().unsqueeze(-1) * n, torch.sign(R_s).unsqueeze(-1) * torch.stack([n[..., 1], -n[..., 0]], -1)


def _soft_below(value, cut, width):
    """1 where value < cut: a step (width 0) or sigmoid((cut - value) / width); NaN -> 0 (with a zero gradient)."""
    finite = ~torch.isnan(value)
    value = torch.where(finite, value, torch.as_tensor(cut, dtype=value.dtype, device=value.device))
    soft = (value < cut).to(value.dtype) if width == 0 else torch.sigmoid((cut - value) / width)
    return torch.where(finite, soft, 0.0)


def _soft_above(value, cut, width):
    """1 where value > cut (as _soft_below)."""
    return _soft_below(-value, -cut, width)


def _expand(mask, like):
    """mask viewed with trailing singleton dimensions to broadcast against like."""
    return mask.view(*mask.shape, *[1] * (like.dim() - mask.dim()))

# ------------------------------------------------------------------------------------------------
# One track's helix (perigee p = (d0, z0, phi, theta, q/p) at the beam line, z reference 0), vmapped below
# ------------------------------------------------------------------------------------------------

def _helix_radius(p, bz):
    """Signed radius rho = sin(theta) / (q/p K Bz) (cm; > 0: clockwise seen from +z); p (..., 5)."""
    return torch.sin(p[..., 3]) / (p[..., 4] * K_CM * bz)


def _helix_centre(p, bz, beam):
    """The helix's circle centre in x-y (..., 2) and its signed radius (...); p (..., 5)."""
    rho, phi = _helix_radius(p, bz), p[..., 2]
    return (beam + p[..., 0].unsqueeze(-1) * torch.stack([-torch.sin(phi), torch.cos(phi)], -1)
            + rho.unsqueeze(-1) * torch.stack([torch.sin(phi), -torch.cos(phi)], -1)), rho


def _helix_at(p, rho, centre, xy):
    """The helix at the x-y point xy (..., 2): its position (..., 3) (z by arc length) and unit direction."""
    _, t = _circle_point(centre, rho, xy)
    z = p[..., 1] - rho * _wrap(torch.atan2(t[..., 1], t[..., 0]) - p[..., 2]) / torch.tan(p[..., 3])
    return (torch.cat([xy, z.unsqueeze(-1)], -1),
            torch.cat([torch.sin(p[..., 3]).unsqueeze(-1) * t, torch.cos(p[..., 3]).unsqueeze(-1)], -1))


def _perigee_at(p, bz, beam, point):
    """The helix's perigee parameters (5,) at the line along z through point (3,) (z0 relative to point), and its
    point of closest approach in x-y (3,)."""
    z0, phi, theta = p[1], p[2], p[3]
    centre, rho = _helix_centre(p, bz, beam)
    u = point[:2] - centre
    u = u / u.norm()
    pca = centre + rho.abs() * u
    phi_new = torch.atan2(-torch.sign(rho) * u[0], torch.sign(rho) * u[1])
    z = z0 - rho * _wrap(phi_new - phi) / torch.tan(theta)          # z0 + arc length x cot(theta)
    d0_new = (pca - point[:2]) @ torch.stack([-torch.sin(phi_new), torch.cos(phi_new)])
    new = torch.stack([d0_new, z - point[2], phi_new, theta, p[4]])
    return new, torch.cat([pca, z.unsqueeze(0)])


def _impact_3d(p, bz, beam, point, iterations=20, guarded=False):
    """The helix's 3D point of closest approach to point (Newton in the turning angle from the perigee, as ACTS), as
    local coordinates (loc0, loc1) on the curvilinear plane there (axes U = z x T / |.|, V = T x U, T the direction),
    and U, V. guarded: also Newton from the transverse closest approach on the three turns nearest the point in z;
    the closest distance minimum wins (far points, low pT)."""
    d0, z0, phi0, theta = p[0], p[1], p[2], p[3]
    rho, cot = _helix_radius(p, bz), 1.0 / torch.tan(p[3])
    centre = torch.stack([beam[0] - (d0 - rho) * torch.sin(phi0), beam[1] + (d0 - rho) * torch.cos(phi0),
                          z0 + rho * phi0 * cot])
    dx, dy, dz = point - centre

    def step(phi):
        first = rho * (dx * torch.cos(phi) + dy * torch.sin(phi) + (dz + rho * phi * cot) * cot)
        second = rho * (-dx * torch.sin(phi) + dy * torch.cos(phi) + rho * cot * cot)
        return phi - first / second

    def root(phi):
        with torch.no_grad():
            for _ in range(iterations - 1):
                phi = step(phi)
        return phi.detach()
    phi = root(phi0)
    if guarded:
        distance = lambda a: (centre + rho * torch.stack([-torch.sin(a), torch.cos(a), -cot * a]) - point).norm()  # noqa: E731
        minimum = lambda a: rho * (-dx * torch.sin(a) + dy * torch.cos(a) + rho * cot * cot) > 0  # noqa: E731
        sign = torch.sign(rho)
        transverse = torch.atan2(-sign * dx, sign * dy)
        steep = cot.abs() > 1e-6
        along_z = torch.where(steep, -dz / torch.where(steep, rho * cot, 1.0), phi0)
        turn = torch.round((along_z - transverse) / (2 * math.pi))
        for k in (-1, 0, 1):
            other = root(transverse + 2 * math.pi * (turn + k))
            phi = torch.where(minimum(other) & (distance(other) < distance(phi) * (1 - 1e-9)), other, phi)
    # the root without derivatives, then one step with them: at a Newton root that step's derivative is the root's
    phi = step(phi)
    position = centre + rho * torch.stack([-torch.sin(phi), torch.cos(phi), -cot * phi])
    direction = torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)])
    axis = torch.where(direction[2].abs() < _CURVILINEAR_TOLERANCE, direction.new_tensor([0.0, 0.0, 1.0]),
                       direction.new_tensor([1.0, 0.0, 0.0]))
    U = torch.linalg.cross(axis, direction)
    U = U / U.norm()
    V = torch.linalg.cross(direction, U)
    offset = position - point
    return torch.stack([offset @ U, offset @ V]), torch.stack([U, V])


def _with_jacobian(function):
    """function(p, bz, beam, point) -> (out, aux) batched over tracks, with d out / d p."""
    def value_and_aux(p, bz, beam, point):
        out, aux = function(p, bz, beam, point)
        return out, (out, aux)
    batched = vmap(jacrev(value_and_aux, has_aux=True), in_dims=(0, 0, None, 0))

    def run(p, bz, beam, point):
        shape = p.shape[:-1]
        jac, (out, aux) = batched(p.reshape(-1, 5), bz.reshape(-1), beam, point.expand(*shape, 3).reshape(-1, 3))
        return (out.reshape(*shape, *out.shape[1:]), aux.reshape(*shape, *aux.shape[1:]),
                jac.reshape(*shape, *jac.shape[1:]))
    return run


def _without_jacobian(function):
    """function(p, bz, beam, point) -> (out, aux) batched over tracks."""
    batched = vmap(function, in_dims=(0, 0, None, 0))

    def run(p, bz, beam, point):
        shape = p.shape[:-1]
        out, aux = batched(p.reshape(-1, 5), bz.reshape(-1), beam, point.expand(*shape, 3).reshape(-1, 3))
        return out.reshape(*shape, *out.shape[1:]), aux.reshape(*shape, *aux.shape[1:])
    return run


_perigee_at_points = _without_jacobian(_perigee_at)
_impact_at_points = _without_jacobian(_impact_3d)
_perigee_with_jacobian = _with_jacobian(_perigee_at)
_impact_with_jacobian = _with_jacobian(_impact_3d)
_guarded_impact_with_jacobian = _with_jacobian(functools.partial(_impact_3d, guarded=True))


def _transport(perigee, covariance, bz, beam, point):
    """Perigee parameters (..., 5) and covariance (..., 5, 5) at the line along z through point (..., 3), and the
    x-y point of closest approach (..., 3); perigee, covariance, bz (...,) at the beam line."""
    new, pca, J = _perigee_with_jacobian(perigee, bz, beam, point)
    return new, J @ covariance @ J.transpose(-1, -2), pca


def _linearise(perigee, covariance, bz, beam, point, transport=None):
    """
    The tracks linearised at point (..., 3) (Piacquadio's helix Jacobians, as ACTS's HelicalTrackLinearizer):
    perigee parameters q(x, p) ~ c + A x + B p of the vertex x and the momentum p = (phi, theta, q/p) there.
    Returns A, B (..., 5, 3), q - c (..., 5) at the linearisation and the weight G = cov^-1 (..., 5, 5).
    transport: the tracks already transported to point (_transport's).
    """
    q, cov, pca = _transport(perigee, covariance, bz, beam, point) if transport is None else transport
    phi, theta = q[..., 2], q[..., 3]
    rho = _helix_radius(q, bz)
    h = torch.where(rho < 0, -1.0, 1.0).to(rho.dtype)
    X = pca[..., 0] - point[..., 0] + rho * torch.sin(phi)
    Y = pca[..., 1] - point[..., 1] - rho * torch.cos(phi)
    S2 = X * X + Y * Y
    S = S2.sqrt()
    rho_cot, abs_rho_over_s = rho / torch.tan(theta), h * rho / S
    zero, one = torch.zeros_like(X), torch.ones_like(X)
    A = torch.stack([torch.stack([-h * X / S, -h * Y / S, zero], -1),
                     torch.stack([rho_cot * Y / S2, -rho_cot * X / S2, one], -1),
                     torch.stack([-Y / S2, X / S2, zero], -1),
                     torch.stack([zero, zero, zero], -1), torch.stack([zero, zero, zero], -1)], -2)
    B = torch.stack([torch.stack([zero, zero, zero], -1), torch.stack([rho_cot * (1 - abs_rho_over_s), zero, zero], -1),
                     torch.stack([abs_rho_over_s, zero, zero], -1), torch.stack([zero, one, zero], -1),
                     torch.stack([zero, zero, one], -1)], -2)
    momentum = torch.stack([phi, theta, q[..., 4]], -1)
    # q - c = A x_pca + B p (ACTS's constant term c = q - A x_pca - B p)
    q_minus_c = (A @ pca.unsqueeze(-1) + B @ momentum.unsqueeze(-1)).squeeze(-1)
    return A, B, q_minus_c, torch.linalg.inv(cov)
