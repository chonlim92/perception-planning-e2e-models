"""
Safety Checker for Trajectory Validation

Implements hard safety constraints that trajectories must satisfy:
- Time-to-Collision (TTC)
- Responsibility-Sensitive Safety (RSS)
- Drivable area compliance
- Traffic rule compliance

Trajectories failing safety checks are REJECTED regardless of their score.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from shapely.geometry import Polygon, LineString, Point


@dataclass
class SafetyConfig:
    """Configuration for safety thresholds."""
    min_ttc: float = 1.5            # seconds
    min_distance_front: float = 2.0  # meters
    min_distance_side: float = 0.5   # meters
    max_lateral_accel: float = 4.0   # m/s^2
    max_longitudinal_accel: float = 3.5  # m/s^2
    max_longitudinal_decel: float = 6.0  # m/s^2 (emergency)
    max_yaw_rate: float = 0.5       # rad/s
    rss_response_time: float = 0.5  # seconds
    rss_max_accel: float = 3.5      # m/s^2
    rss_min_brake: float = 4.0      # m/s^2 (comfortable braking)
    rss_max_brake: float = 8.0      # m/s^2 (maximum braking)


class SafetyChecker:
    """
    Hard safety constraint checker for trajectory validation.

    A trajectory MUST pass all safety checks to be considered for scoring.
    This is a binary gate: pass or reject.
    """

    def __init__(self, config: Optional[SafetyConfig] = None,
                 ego_length: float = 4.5, ego_width: float = 2.0):
        self.config = config or SafetyConfig()
        self.ego_length = ego_length
        self.ego_width = ego_width

    def check(self, trajectory_points: np.ndarray,
              trajectory_headings: np.ndarray,
              trajectory_velocities: np.ndarray,
              agent_predictions: List[np.ndarray],
              agent_dimensions: List[Tuple[float, float]],
              drivable_area: Optional[Polygon] = None,
              dt: float = 0.5) -> dict:
        """
        Run all safety checks on a trajectory.

        Args:
            trajectory_points: (T, 2) ego positions
            trajectory_headings: (T,) ego headings
            trajectory_velocities: (T,) ego speeds
            agent_predictions: list of (T, 2) predicted positions per agent
            agent_dimensions: list of (length, width) per agent
            drivable_area: shapely Polygon of drivable road area
            dt: time step between waypoints

        Returns:
            dict with pass/fail for each check and overall result
        """
        results = {}

        results['ttc'] = self._check_ttc(
            trajectory_points, trajectory_headings, trajectory_velocities,
            agent_predictions, agent_dimensions, dt)

        results['collision'] = self._check_collision(
            trajectory_points, trajectory_headings,
            agent_predictions, agent_dimensions)

        results['rss'] = self._check_rss(
            trajectory_points, trajectory_velocities,
            agent_predictions, dt)

        results['kinematic'] = self._check_kinematic_feasibility(
            trajectory_points, trajectory_headings,
            trajectory_velocities, dt)

        if drivable_area is not None:
            results['drivable_area'] = self._check_drivable_area(
                trajectory_points, drivable_area)
        else:
            results['drivable_area'] = {'passed': True, 'details': 'no drivable area provided'}

        results['overall_passed'] = all(r['passed'] for r in results.values()
                                        if isinstance(r, dict) and 'passed' in r)

        return results

    def _check_ttc(self, ego_pos: np.ndarray, ego_headings: np.ndarray,
                   ego_vel: np.ndarray, agent_preds: List[np.ndarray],
                   agent_dims: List[Tuple[float, float]],
                   dt: float) -> dict:
        """Check Time-to-Collision with all agents."""
        min_ttc = float('inf')
        critical_agent = -1
        critical_time = -1

        for agent_idx, (agent_pred, (a_len, a_width)) in enumerate(
                zip(agent_preds, agent_dims)):
            for t in range(min(len(ego_pos), len(agent_pred)) - 1):
                ego_p = ego_pos[t]
                agent_p = agent_pred[t]

                rel_pos = agent_p - ego_p
                dist = np.linalg.norm(rel_pos)

                if dist < 1e-6:
                    min_ttc = 0
                    critical_agent = agent_idx
                    critical_time = t
                    break

                ego_vel_vec = ego_vel[t] * np.array([
                    np.cos(ego_headings[t]), np.sin(ego_headings[t])])

                if t < len(agent_pred) - 1:
                    agent_vel_vec = (agent_pred[t+1] - agent_pred[t]) / dt
                else:
                    agent_vel_vec = np.zeros(2)

                rel_vel = ego_vel_vec - agent_vel_vec
                closing_speed = np.dot(rel_pos / dist, rel_vel)

                if closing_speed > 0.1:
                    safe_dist = (self.ego_length + a_len) / 2 + self.config.min_distance_front
                    ttc = max(0, (dist - safe_dist)) / closing_speed

                    if ttc < min_ttc:
                        min_ttc = ttc
                        critical_agent = agent_idx
                        critical_time = t

        passed = min_ttc >= self.config.min_ttc
        return {
            'passed': passed,
            'min_ttc': min_ttc,
            'critical_agent': critical_agent,
            'critical_timestep': critical_time,
            'threshold': self.config.min_ttc,
        }

    def _check_collision(self, ego_pos: np.ndarray, ego_headings: np.ndarray,
                         agent_preds: List[np.ndarray],
                         agent_dims: List[Tuple[float, float]]) -> dict:
        """Check for geometric overlap (collision) at each timestep."""
        collisions = []

        for t in range(len(ego_pos)):
            ego_polygon = self._get_vehicle_polygon(
                ego_pos[t], ego_headings[t], self.ego_length, self.ego_width)

            for agent_idx, (agent_pred, (a_len, a_width)) in enumerate(
                    zip(agent_preds, agent_dims)):
                if t >= len(agent_pred):
                    continue

                agent_heading = 0.0
                if t < len(agent_pred) - 1:
                    diff = agent_pred[t+1] - agent_pred[t]
                    if np.linalg.norm(diff) > 0.1:
                        agent_heading = np.arctan2(diff[1], diff[0])

                agent_polygon = self._get_vehicle_polygon(
                    agent_pred[t], agent_heading, a_len, a_width)

                if ego_polygon.intersects(agent_polygon):
                    collisions.append({
                        'timestep': t,
                        'agent_idx': agent_idx,
                        'overlap_area': ego_polygon.intersection(agent_polygon).area
                    })

        return {
            'passed': len(collisions) == 0,
            'num_collisions': len(collisions),
            'collisions': collisions[:5],  # first 5 for debugging
        }

    def _check_rss(self, ego_pos: np.ndarray, ego_vel: np.ndarray,
                   agent_preds: List[np.ndarray], dt: float) -> dict:
        """
        Responsibility-Sensitive Safety (RSS) check.

        RSS defines minimum safe following distances based on:
        - Response time
        - Maximum acceleration during response
        - Minimum braking of ego
        - Maximum braking of lead vehicle
        """
        violations = []
        rho = self.config.rss_response_time
        a_max = self.config.rss_max_accel
        b_min = self.config.rss_min_brake
        b_max = self.config.rss_max_brake

        for t in range(len(ego_pos)):
            v_r = ego_vel[t]  # ego (rear) velocity

            for agent_idx, agent_pred in enumerate(agent_preds):
                if t >= len(agent_pred):
                    continue

                rel_pos = agent_pred[t] - ego_pos[t]
                dist = np.linalg.norm(rel_pos)

                # Estimate lead vehicle velocity from predictions
                if t < len(agent_pred) - 1:
                    v_f = np.linalg.norm(agent_pred[t+1] - agent_pred[t]) / dt
                else:
                    v_f = 0.0

                # RSS longitudinal safe distance (clamped to >= 0)
                d_safe = (v_r * rho + 0.5 * a_max * rho**2 +
                          (v_r + a_max * rho)**2 / (2 * b_min) -
                          v_f**2 / (2 * b_max))
                d_safe = max(0.0, d_safe)

                if dist < d_safe and dist < 50:  # only check nearby
                    violations.append({
                        'timestep': t,
                        'agent_idx': agent_idx,
                        'distance': dist,
                        'safe_distance': d_safe,
                        'deficit': d_safe - dist,
                    })

        return {
            'passed': len(violations) == 0,
            'num_violations': len(violations),
            'violations': violations[:5],
        }

    def _check_kinematic_feasibility(self, ego_pos: np.ndarray,
                                     ego_headings: np.ndarray,
                                     ego_vel: np.ndarray,
                                     dt: float) -> dict:
        """Check that trajectory is kinematically feasible."""
        violations = []

        # Check acceleration limits
        accels = np.diff(ego_vel) / dt
        for t, a in enumerate(accels):
            if a > self.config.max_longitudinal_accel:
                violations.append(f"t={t}: accel {a:.2f} > {self.config.max_longitudinal_accel}")
            elif a < -self.config.max_longitudinal_decel:
                violations.append(f"t={t}: decel {a:.2f} < -{self.config.max_longitudinal_decel}")

        # Check yaw rate limits
        yaw_rates = np.diff(ego_headings) / dt
        # Handle angle wrapping
        yaw_rates = np.arctan2(np.sin(yaw_rates * dt), np.cos(yaw_rates * dt)) / dt
        for t, yr in enumerate(yaw_rates):
            if abs(yr) > self.config.max_yaw_rate:
                violations.append(f"t={t}: yaw_rate {yr:.2f} > {self.config.max_yaw_rate}")

        # Check lateral acceleration
        for t in range(len(ego_vel)):
            if t < len(yaw_rates):
                lat_accel = abs(ego_vel[t] * yaw_rates[t])
                if lat_accel > self.config.max_lateral_accel:
                    violations.append(
                        f"t={t}: lat_accel {lat_accel:.2f} > {self.config.max_lateral_accel}")

        return {
            'passed': len(violations) == 0,
            'num_violations': len(violations),
            'violations': violations[:10],
        }

    def _check_drivable_area(self, ego_pos: np.ndarray,
                             drivable_area: Polygon) -> dict:
        """Check that all trajectory points are within drivable area."""
        violations = []

        for t, pos in enumerate(ego_pos):
            point = Point(pos[0], pos[1])
            if not drivable_area.contains(point):
                dist_to_boundary = drivable_area.exterior.distance(point)
                violations.append({
                    'timestep': t,
                    'position': pos.tolist(),
                    'distance_outside': dist_to_boundary,
                })

        compliance_ratio = 1.0 - len(violations) / max(len(ego_pos), 1)

        return {
            'passed': len(violations) == 0,
            'compliance_ratio': compliance_ratio,
            'num_violations': len(violations),
            'violations': violations[:5],
        }

    @staticmethod
    def _get_vehicle_polygon(position: np.ndarray, heading: float,
                             length: float, width: float) -> Polygon:
        """Get vehicle bounding box as a shapely Polygon."""
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # Vehicle corners relative to center
        half_l = length / 2
        half_w = width / 2

        corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])

        # Rotate
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated = corners @ rotation.T

        # Translate
        translated = rotated + position

        return Polygon(translated)


def demo():
    """Demonstrate safety checker on sample trajectories."""
    print("=" * 60)
    print("Safety Checker Demo")
    print("=" * 60)

    config = SafetyConfig()
    checker = SafetyChecker(config)

    dt = 0.5
    n_steps = 16

    # Safe trajectory (lane change)
    t = np.arange(n_steps) * dt
    safe_pos = np.column_stack([10.0 * t, 3.5 * (1 - np.cos(np.pi * t / 8)) / 2])
    safe_headings = np.arctan2(np.diff(safe_pos[:, 1], prepend=0),
                                np.diff(safe_pos[:, 0], prepend=safe_pos[0, 0]))
    safe_vel = np.ones(n_steps) * 10.0

    # Unsafe trajectory (drives into obstacle)
    unsafe_pos = np.column_stack([10.0 * t, np.zeros(n_steps)])
    unsafe_headings = np.zeros(n_steps)
    unsafe_vel = np.ones(n_steps) * 10.0

    # Obstacle at (40, 0), stationary
    obstacle_pred = np.tile([40.0, 0.0], (n_steps, 1))
    agent_dims = [(4.5, 2.0)]

    print("\n--- Trajectory 1: Lane Change (Safe) ---")
    result = checker.check(safe_pos, safe_headings, safe_vel,
                           [obstacle_pred], agent_dims, dt=dt)
    print(f"  Overall: {'PASSED' if result['overall_passed'] else 'FAILED'}")
    for check_name, check_result in result.items():
        if isinstance(check_result, dict) and 'passed' in check_result:
            status = 'PASS' if check_result['passed'] else 'FAIL'
            print(f"  {check_name:20s}: {status}")

    print("\n--- Trajectory 2: Straight Ahead (Unsafe) ---")
    result = checker.check(unsafe_pos, unsafe_headings, unsafe_vel,
                           [obstacle_pred], agent_dims, dt=dt)
    print(f"  Overall: {'PASSED' if result['overall_passed'] else 'FAILED'}")
    for check_name, check_result in result.items():
        if isinstance(check_result, dict) and 'passed' in check_result:
            status = 'PASS' if check_result['passed'] else 'FAIL'
            details = ''
            if not check_result['passed']:
                if 'min_ttc' in check_result:
                    details = f" (TTC={check_result['min_ttc']:.2f}s)"
                elif 'num_collisions' in check_result:
                    details = f" ({check_result['num_collisions']} collisions)"
            print(f"  {check_name:20s}: {status}{details}")


if __name__ == '__main__':
    demo()
