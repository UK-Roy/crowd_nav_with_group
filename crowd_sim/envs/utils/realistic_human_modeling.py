"""Realistic pedestrian / group modeling utilities.

Features are phased in; each phase is gated behind a config flag so that
disabling every flag reproduces the legacy environment bit-exactly.

Implemented phases:
- A: individual preferred-speed sampling (Weidmann 1992)
- B: group speed factor (Moussaid et al. 2010)
- C: F-formations for static groups (Kendon 1990)
- D: leader-follower dynamics for dynamic groups (Helbing & Molnar 1995)

References:
- Weidmann, U. (1992). Transporttechnik der Fussganger. ETH Zurich, IVT.
- Moussaid, M. et al. (2010). The walking behaviour of pedestrian social groups.
- Kendon, A. (1990). Conducting Interaction.
- Helbing, D. & Molnar, P. (1995). Social force model for pedestrian dynamics.
"""

import numpy as np
try:
    from scipy.spatial import ConvexHull, Delaunay, QhullError
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ---------------------------------------------------------------------------
# Phase E — ConvexHull group geometry
# ---------------------------------------------------------------------------

def _collinear(pts, tol=1e-6):
    """True if all points lie on a single line."""
    if len(pts) < 3:
        return True
    v = pts[1] - pts[0]
    for p in pts[2:]:
        cross = v[0] * (p[1] - pts[0][1]) - v[1] * (p[0] - pts[0][0])
        if abs(cross) > tol:
            return False
    return True


def _rect_half_planes(a, b, buf):
    """Half-plane normals and offsets for an oriented rectangle around segment a→b.

    Returns list of (normal, offset) where the inside is normal·x <= offset.
    """
    seg = b - a
    seg_len = np.linalg.norm(seg) + 1e-9
    t = seg / seg_len          # unit tangent
    n = np.array([-t[1], t[0]])  # unit normal (perpendicular)
    mid = (a + b) / 2.0
    planes = [
        ( t,  np.dot( t, b) + 1e-3),   # far cap
        (-t, -np.dot( t, a) + 1e-3),   # near cap
        ( n,  np.dot( n, mid) + buf),   # right side
        (-n, -np.dot( n, mid) + buf),   # left side
    ]
    return planes


class ConvexHullGeometry:
    """Represents a group's spatial boundary with degeneracy handling.

    n=1 (singleton)  → circle of radius human_radius
    n=2 or collinear → oriented rectangle (segment ± hull_degenerate_buffer)
    n>=3 generic     → scipy ConvexHull; point-in-hull via Delaunay

    All policies can call:
        hull.contains(point)        → bool
        hull.bounding_radius        → scalar (largest vertex distance from centroid)
        hull.centroid               → np.ndarray shape (2,)
    """

    def __init__(self, positions, human_radius=0.3, buffer=0.30):
        """
        positions: (n, 2) array of member positions
        buffer:    half-width for rectangle degenerate fallback (m)
        """
        positions = np.asarray(positions, dtype=float)
        n = len(positions)
        self._buffer = buffer
        self._human_radius = human_radius
        self.centroid = positions.mean(axis=0)

        if n == 1 or (not _SCIPY_OK) or n < 3 or _collinear(positions):
            if n == 1:
                self._kind = "circle"
                self._center = positions[0].copy()
                self._radius = human_radius + buffer
            else:
                self._kind = "rectangle"
                self._a = positions[0].copy()
                self._b = positions[-1].copy()
                self._planes = _rect_half_planes(self._a, self._b, buffer)
        else:
            try:
                hull = ConvexHull(positions)
                verts = positions[hull.vertices]
                # Expand by buffer: push each vertex outward from centroid
                dirs = verts - self.centroid
                norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
                verts_expanded = verts + (dirs / norms) * buffer
                self._kind = "polygon"
                self._delaunay = Delaunay(verts_expanded)
                self._verts = verts_expanded
            except Exception:
                # Fallback to rectangle if hull fails
                self._kind = "rectangle"
                self._a = positions[0].copy()
                self._b = positions[-1].copy()
                self._planes = _rect_half_planes(self._a, self._b, buffer)

        # Bounding radius: max distance from centroid to any vertex
        if self._kind == "circle":
            self.bounding_radius = self._radius
        elif self._kind == "rectangle":
            corners = np.array([self._a, self._b])
            self.bounding_radius = float(np.max(
                np.linalg.norm(corners - self.centroid, axis=1))) + buffer
        else:
            self.bounding_radius = float(np.max(
                np.linalg.norm(self._verts - self.centroid, axis=1)))

    def contains(self, point):
        """Return True if point (array-like len 2) is inside the hull."""
        p = np.asarray(point, dtype=float)
        if self._kind == "circle":
            return bool(np.linalg.norm(p - self._center) <= self._radius)
        if self._kind == "rectangle":
            return all(np.dot(n, p) <= off for n, off in self._planes)
        return bool(self._delaunay.find_simplex(p) >= 0)

    def distance_to_boundary(self, point):
        """Signed distance: negative = inside, positive = outside.
        Uses bounding_radius approximation for speed.
        """
        p = np.asarray(point, dtype=float)
        d = float(np.linalg.norm(p - self.centroid)) - self.bounding_radius
        return d


def build_group_hulls(group_positions_dict, human_radius=0.3, buffer=0.30):
    """Build a ConvexHullGeometry per group from a {group_id: positions} dict."""
    return {gid: ConvexHullGeometry(pos, human_radius, buffer)
            for gid, pos in group_positions_dict.items() if len(pos) > 0}


# ---------------------------------------------------------------------------
# Phase B — group speed factor
# ---------------------------------------------------------------------------

def group_target_speed(member_speeds, cfg):
    """Moussaid et al. (2010): groups walk at factor x slowest member speed."""
    return cfg.realistic.group_speed_factor * min(member_speeds)


# ---------------------------------------------------------------------------
# Phase A — individual speed distribution
# ---------------------------------------------------------------------------

def sample_individual_speed(cfg, rng=None):
    """Weidmann (1992): v ~ N(1.34, 0.26) clipped to [0.8, 1.8] m/s."""
    r = cfg.realistic
    if rng is None:
        v = np.random.normal(r.individual_speed_mean, r.individual_speed_std)
    else:
        v = rng.normal(r.individual_speed_mean, r.individual_speed_std)
    return float(np.clip(v, r.individual_speed_min, r.individual_speed_max))


# ---------------------------------------------------------------------------
# Phase C — F-formations for static groups (Kendon 1990)
# ---------------------------------------------------------------------------

def _f_formation_positions(n_members, centroid, radius, rng):
    """Return (px, py, theta) for each member in a Kendon F-formation.

    Formation is picked randomly from those applicable to n_members:
      n=2 -> vis-a-vis, L-shape, side-by-side
      n>=3 -> circle (o-space ring)

    centroid: (cx, cy) in world frame
    radius:   distance from centroid to each member (f_formation_radius)
    Returns list of (px, py, theta) where theta faces the o-space centre.
    """
    cx, cy = centroid

    if n_members == 2:
        formation = rng.choice(["vis_a_vis", "l_shape", "side_by_side"])
    else:
        formation = "circle"

    # Random group orientation in the world (global rotation)
    base_angle = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(base_angle), -np.sin(base_angle)],
                  [np.sin(base_angle),  np.cos(base_angle)]])

    if formation == "vis_a_vis":
        # Two members facing each other across the o-space
        local = np.array([[ radius, 0.0],
                          [-radius, 0.0]])
        thetas = [base_angle + np.pi, base_angle]          # face each other

    elif formation == "l_shape":
        # 90-degree arrangement
        local = np.array([[radius, 0.0],
                          [0.0,   radius]])
        thetas = [base_angle + np.pi,
                  base_angle + 3 * np.pi / 2]

    elif formation == "side_by_side":
        # Both face the same direction, shoulder to shoulder
        local = np.array([[ radius / 2, 0.0],
                          [-radius / 2, 0.0]])
        thetas = [base_angle + np.pi / 2,
                  base_angle + np.pi / 2]

    else:  # circle — evenly spaced, each facing inward
        angles = np.linspace(0, 2 * np.pi, n_members, endpoint=False)
        local = np.stack([radius * np.cos(angles),
                          radius * np.sin(angles)], axis=1)
        thetas = [a + np.pi for a in angles]   # face centroid

    world = (R @ local.T).T + np.array([cx, cy])
    return [(float(world[i, 0]), float(world[i, 1]), float(thetas[i]))
            for i in range(n_members)]


def apply_f_formation(grp, robot, humans, cfg, rng=None):
    """Position a static group's members using an F-formation.

    Replaces Group.position_members for static groups when
    realistic.use_f_formations is True. Returns the updated humans list.
    """
    if rng is None:
        rng = np.random.default_rng()

    radius = cfg.realistic.f_formation_radius
    cx, cy = grp.centroid

    poses = _f_formation_positions(len(grp.members), (cx, cy), radius, rng)
    all_agents = [robot] + humans
    gap = 0.15  # extra clearance beyond sum-of-radii

    for (px, py, theta), mem in zip(poses, grp.members):
        # Jitter to resolve spawn collisions using per-pair radius sum
        for _ in range(300):
            collision = any(
                np.linalg.norm([px - a.px, py - a.py]) < (mem.radius + a.radius + gap)
                for a in all_agents if a.px is not None
            )
            if not collision:
                break
            px += rng.uniform(-0.4, 0.4)
            py += rng.uniform(-0.4, 0.4)

        # goal = mirror of position through origin (circle-crossing convention)
        mem.set(px, py, -px, -py, 0.0, 0.0, theta)
        humans.append(mem)
        all_agents.append(mem)

    grp.positioned = True
    return humans


# ---------------------------------------------------------------------------
# Phase D — leader-follower dynamics for dynamic groups
# ---------------------------------------------------------------------------

def leader_follower_action(follower, leader, cfg, rank=0, siblings=None):
    """Compute follower velocity command blending follow-force + social force.

    rank:     follower index (0 = first follower, 1 = second, ...).
              Used to assign a unique staggered lateral slot so siblings
              don't converge to the same point.
    siblings: list of other Human agents in the same group (for inter-follower
              repulsion); pass [] or None to skip.

    Returns (vx, vy) clipped to follower.v_pref.
    """
    r = cfg.realistic
    ts = cfg.env.time_step
    A = cfg.sf.A
    B = cfg.sf.B

    spacing = r.leader_follower_spacing
    lx, ly = leader.px, leader.py
    heading = np.arctan2(leader.vy, leader.vx) if (leader.vx != 0 or leader.vy != 0) else 0.0

    # Unit vectors along and perpendicular to leader heading
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    perp_x, perp_y = -sin_h, cos_h  # left-perpendicular

    # Each follower gets a unique lateral slot: 0, +1, -1, +2, -2, ...
    slot = (rank + 1) // 2 * (1 if rank % 2 == 1 else -1)
    lateral_offset = slot * (2 * follower.radius + 0.1)

    target_x = lx - spacing * cos_h + lateral_offset * perp_x
    target_y = ly - spacing * sin_h + lateral_offset * perp_y

    # Follow force (proportional controller)
    fx = r.leader_follower_gain * (target_x - follower.px)
    fy = r.leader_follower_gain * (target_y - follower.py)

    # Social repulsion from all group members (leader + siblings)
    for agent in ([leader] + (siblings or [])):
        if agent is follower:
            continue
        dx = follower.px - agent.px
        dy = follower.py - agent.py
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        overlap = follower.radius + agent.radius - dist
        rep = A * np.exp(overlap / B)
        fx += rep * dx / dist
        fy += rep * dy / dist

    # Integrate and clip to v_pref
    vx = follower.vx + (fx + (0.0 - follower.vx) / (cfg.sf.KI + 1e-6)) * ts
    vy = follower.vy + (fy + (0.0 - follower.vy) / (cfg.sf.KI + 1e-6)) * ts
    speed = np.sqrt(vx * vx + vy * vy)
    if speed > follower.v_pref:
        vx = vx / speed * follower.v_pref
        vy = vy / speed * follower.v_pref
    return float(vx), float(vy)
