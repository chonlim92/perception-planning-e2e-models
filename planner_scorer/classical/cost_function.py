"""
Classical Weighted Multi-Criteria Trajectory Scorer

Scores candidate trajectories using a weighted combination of interpretable
cost functions: safety (collision risk), comfort (jerk/acceleration),
progress (route efficiency), and rule compliance (lane keeping, speed limits).

Usage:
    python cost_function.py --demo
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import argparse


@dataclass
class TrajectoryPoint:
    """Single waypoint in a trajectory."""
    x: float          # meters, ego-centric
    y: float          # meters, ego-centric
    heading: float    # radians
    velocity: float   # m/s
    acceleration: float = 0.0  # m/s^2
    curvature: float = 0.0     # 1/m


@dataclass
class Trajectory:
    """A candidate trajectory as a sequence of waypoints."""
    points: List[TrajectoryPoint]
    dt: float = 0.5  # time between waypoints in seconds

    @property
    def duration(self) -> float:
        return len(self.points) * self.dt

    @property
    def positions(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.points])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([p.velocity for p in self.points])

    @property
    def accelerations(self) -> np.ndarray:
        return np.array([p.acceleration for p in self.points])

    @property
    def curvatures(self) -> np.ndarray:
        return np.array([p.curvature for p in self.points])

    @property
    def headings(self) -> np.ndarray:
        return np.array([p.heading for p in self.points])


@dataclass
class AgentState:
    """State of another agent in the scene."""
    x: float
    y: float
    heading: float
    velocity: float
    length: float = 4.5  # meters
    width: float = 2.0   # meters
    predicted_trajectory: Optional[np.ndarray] = None  # (T, 2) future positions


@dataclass
class SceneContext:
    """Full scene context for scoring."""
    agents: List[AgentState] = field(default_factory=list)
    lane_centers: Optional[np.ndarray] = None      # (N, 2) polyline points
    road_boundaries: Optional[np.ndarray] = None   # (N, 2) boundary points
    speed_limit: float = 13.89  # m/s (50 km/h default)
    ego_route: Optional[np.ndarray] = None         # (N, 2) route centerline


@dataclass
class ScorerWeights:
    """Weights for multi-criteria scoring."""
    collision: float = 5.0
    ttc: float = 3.0
    comfort_accel: float = 1.0
    comfort_jerk: float = 1.5
    comfort_curvature: float = 1.0
    progress: float = 2.0
    lane_keeping: float = 1.5
    speed_limit: float = 2.0


class ClassicalTrajectoryScorer:
    """
    Weighted multi-criteria trajectory scorer.

    Computes a composite score S(t) = -Σᵢ wᵢ · cᵢ(t)
    where cᵢ are individual cost terms (lower = better).
    Final score is negated so higher = better trajectory.
    """

    def __init__(self, weights: Optional[ScorerWeights] = None,
                 ego_length: float = 4.5, ego_width: float = 2.0):
        self.weights = weights or ScorerWeights()
        self.ego_length = ego_length
        self.ego_width = ego_width

    def score(self, trajectory: Trajectory, context: SceneContext) -> dict:
        """
        Score a single trajectory given scene context.

        Returns dict with total score and individual sub-scores.
        """
        costs = {}

        costs['collision'] = self._collision_cost(trajectory, context)
        costs['ttc'] = self._ttc_cost(trajectory, context)
        costs['comfort_accel'] = self._acceleration_cost(trajectory)
        costs['comfort_jerk'] = self._jerk_cost(trajectory)
        costs['comfort_curvature'] = self._curvature_cost(trajectory)
        costs['progress'] = self._progress_cost(trajectory, context)
        costs['lane_keeping'] = self._lane_keeping_cost(trajectory, context)
        costs['speed_limit'] = self._speed_limit_cost(trajectory, context)

        total_cost = (
            self.weights.collision * costs['collision'] +
            self.weights.ttc * costs['ttc'] +
            self.weights.comfort_accel * costs['comfort_accel'] +
            self.weights.comfort_jerk * costs['comfort_jerk'] +
            self.weights.comfort_curvature * costs['comfort_curvature'] +
            self.weights.progress * costs['progress'] +
            self.weights.lane_keeping * costs['lane_keeping'] +
            self.weights.speed_limit * costs['speed_limit']
        )

        return {
            'total_score': -total_cost,  # higher = better
            'total_cost': total_cost,
            'sub_costs': costs,
        }

    def score_batch(self, trajectories: List[Trajectory],
                    context: SceneContext) -> List[dict]:
        """Score multiple trajectories and rank them."""
        results = [self.score(t, context) for t in trajectories]
        results.sort(key=lambda r: r['total_score'], reverse=True)
        return results

    def select_best(self, trajectories: List[Trajectory],
                    context: SceneContext) -> Tuple[int, Trajectory, dict]:
        """Select the best trajectory from candidates."""
        scores = [self.score(t, context) for t in trajectories]
        best_idx = max(range(len(scores)), key=lambda i: scores[i]['total_score'])
        return best_idx, trajectories[best_idx], scores[best_idx]

    def _collision_cost(self, trajectory: Trajectory,
                        context: SceneContext) -> float:
        """
        Cost based on minimum distance to other agents.
        Uses simplified circle-based collision checking.
        """
        if not context.agents:
            return 0.0

        ego_radius = np.sqrt((self.ego_length/2)**2 + (self.ego_width/2)**2)
        min_distance = float('inf')

        for t_idx, point in enumerate(trajectory.points):
            ego_pos = np.array([point.x, point.y])

            for agent in context.agents:
                if agent.predicted_trajectory is not None and t_idx < len(agent.predicted_trajectory):
                    agent_pos = agent.predicted_trajectory[t_idx]
                else:
                    agent_pos = np.array([agent.x, agent.y])

                agent_radius = np.sqrt((agent.length/2)**2 + (agent.width/2)**2)
                dist = np.linalg.norm(ego_pos - agent_pos) - ego_radius - agent_radius
                min_distance = min(min_distance, dist)

        if min_distance < 0:
            return 10.0  # collision detected
        elif min_distance < 2.0:
            return (2.0 - min_distance) / 2.0 * 5.0  # near-miss penalty
        return 0.0

    def _ttc_cost(self, trajectory: Trajectory,
                  context: SceneContext) -> float:
        """
        Time-to-collision cost.
        Lower TTC = higher cost (more dangerous).
        """
        if not context.agents:
            return 0.0

        min_ttc = float('inf')

        for t_idx in range(len(trajectory.points) - 1):
            ego_pos = np.array([trajectory.points[t_idx].x,
                                trajectory.points[t_idx].y])
            ego_vel = trajectory.points[t_idx].velocity
            ego_heading = trajectory.points[t_idx].heading
            ego_vel_vec = ego_vel * np.array([np.cos(ego_heading),
                                              np.sin(ego_heading)])

            for agent in context.agents:
                if agent.predicted_trajectory is not None and t_idx < len(agent.predicted_trajectory):
                    agent_pos = agent.predicted_trajectory[t_idx]
                else:
                    agent_pos = np.array([agent.x, agent.y])

                agent_vel_vec = agent.velocity * np.array([
                    np.cos(agent.heading), np.sin(agent.heading)])

                rel_pos = agent_pos - ego_pos
                rel_vel = agent_vel_vec - ego_vel_vec

                dist = np.linalg.norm(rel_pos)
                closing_speed = -np.dot(rel_pos, rel_vel) / (dist + 1e-6)

                if closing_speed > 0:
                    safe_dist = self.ego_length + agent.length
                    ttc = (dist - safe_dist) / closing_speed
                    if ttc > 0:
                        min_ttc = min(min_ttc, ttc)

        if min_ttc < 1.5:
            return (1.5 - min_ttc) / 1.5 * 5.0
        elif min_ttc < 3.0:
            return (3.0 - min_ttc) / 3.0 * 1.0
        return 0.0

    def _acceleration_cost(self, trajectory: Trajectory) -> float:
        """Penalize harsh acceleration/deceleration."""
        accels = trajectory.accelerations
        if len(accels) == 0:
            velocities = trajectory.velocities
            accels = np.diff(velocities) / trajectory.dt

        cost = 0.0
        for a in accels:
            if a > 3.0:   # hard acceleration
                cost += (a - 3.0) ** 2
            elif a < -4.0:  # hard braking
                cost += (a + 4.0) ** 2
        return cost / max(len(accels), 1)

    def _jerk_cost(self, trajectory: Trajectory) -> float:
        """Penalize high jerk (rate of acceleration change)."""
        accels = trajectory.accelerations
        if len(accels) == 0:
            velocities = trajectory.velocities
            accels = np.diff(velocities) / trajectory.dt

        if len(accels) < 2:
            return 0.0

        jerks = np.diff(accels) / trajectory.dt
        cost = np.mean(jerks ** 2)

        jerk_threshold = 2.5  # m/s^3
        excess_jerk = np.maximum(np.abs(jerks) - jerk_threshold, 0)
        cost += np.sum(excess_jerk ** 2) / len(jerks)

        return cost

    def _curvature_cost(self, trajectory: Trajectory) -> float:
        """Penalize high curvature (sharp turns)."""
        curvatures = trajectory.curvatures
        if np.all(curvatures == 0):
            positions = trajectory.positions
            if len(positions) < 3:
                return 0.0
            dx = np.diff(positions[:, 0])
            dy = np.diff(positions[:, 1])
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            denom = (dx[:-1]**2 + dy[:-1]**2)**1.5 + 1e-6
            curvatures = np.abs(ddx * dy[:-1] - ddy * dx[:-1]) / denom

        return np.mean(curvatures ** 2) * 100

    def _progress_cost(self, trajectory: Trajectory,
                       context: SceneContext) -> float:
        """
        Penalize lack of progress along route.
        Lower progress = higher cost.
        """
        if context.ego_route is None:
            positions = trajectory.positions
            forward_progress = positions[-1, 0] - positions[0, 0]
            expected_progress = trajectory.duration * context.speed_limit * 0.8
            return max(0, 1.0 - forward_progress / (expected_progress + 1e-6))

        positions = trajectory.positions
        route = context.ego_route

        start_distances = np.linalg.norm(route - positions[0], axis=1)
        start_idx = np.argmin(start_distances)

        end_distances = np.linalg.norm(route - positions[-1], axis=1)
        end_idx = np.argmin(end_distances)

        if end_idx <= start_idx:
            return 1.0  # no progress or going backwards

        route_length = np.sum(np.linalg.norm(np.diff(route[start_idx:end_idx+1], axis=0), axis=1))
        expected = trajectory.duration * context.speed_limit * 0.7
        progress_ratio = route_length / (expected + 1e-6)

        return max(0, 1.0 - progress_ratio)

    def _lane_keeping_cost(self, trajectory: Trajectory,
                           context: SceneContext) -> float:
        """Penalize deviation from lane center."""
        if context.lane_centers is None:
            return 0.0

        positions = trajectory.positions
        total_deviation = 0.0

        for pos in positions:
            distances = np.linalg.norm(context.lane_centers - pos, axis=1)
            min_dist = np.min(distances)
            total_deviation += min_dist ** 2

        return total_deviation / len(positions)

    def _speed_limit_cost(self, trajectory: Trajectory,
                          context: SceneContext) -> float:
        """Penalize exceeding speed limit."""
        velocities = trajectory.velocities
        excess = np.maximum(velocities - context.speed_limit, 0)
        return np.mean(excess ** 2)


def create_demo_trajectories() -> Tuple[List[Trajectory], SceneContext]:
    """Create sample trajectories and context for demonstration."""
    dt = 0.5
    horizon = 8  # seconds
    n_points = int(horizon / dt)

    # Trajectory 1: Lane change left (good - avoids obstacle)
    traj1_points = []
    for i in range(n_points):
        t = i * dt
        x = 10.0 * t  # constant forward progress
        y = 3.5 * (1 - np.cos(np.pi * t / horizon)) / 2  # smooth lateral
        heading = np.arctan2(3.5 * np.pi * np.sin(np.pi * t / horizon) / (2 * horizon), 10.0)
        velocity = 10.0
        traj1_points.append(TrajectoryPoint(x, y, heading, velocity))
    traj1 = Trajectory(traj1_points, dt)

    # Trajectory 2: Slow down (good - safe but slow)
    traj2_points = []
    for i in range(n_points):
        t = i * dt
        velocity = max(2.0, 10.0 - 1.5 * t)
        x = sum([max(2.0, 10.0 - 1.5 * j * dt) * dt for j in range(i)])
        y = 0.0
        heading = 0.0
        accel = -1.5 if velocity > 2.0 else 0.0
        traj2_points.append(TrajectoryPoint(x, y, heading, velocity, accel))
    traj2 = Trajectory(traj2_points, dt)

    # Trajectory 3: Maintain speed (bad - will collide)
    traj3_points = []
    for i in range(n_points):
        t = i * dt
        x = 10.0 * t
        y = 0.0
        heading = 0.0
        velocity = 10.0
        traj3_points.append(TrajectoryPoint(x, y, heading, velocity))
    traj3 = Trajectory(traj3_points, dt)

    # Trajectory 4: Aggressive swerve (bad - uncomfortable)
    traj4_points = []
    for i in range(n_points):
        t = i * dt
        x = 10.0 * t
        y = 4.0 * np.sin(2 * np.pi * t / 3)  # oscillating
        heading = np.arctan2(4.0 * 2 * np.pi / 3 * np.cos(2 * np.pi * t / 3), 10.0)
        velocity = 10.0
        curvature = abs(4.0 * (2*np.pi/3)**2 * np.sin(2*np.pi*t/3)) / (10.0**2 + 1e-6)
        traj4_points.append(TrajectoryPoint(x, y, heading, velocity, curvature=curvature))
    traj4 = Trajectory(traj4_points, dt)

    # Scene context: obstacle ahead in current lane
    obstacle = AgentState(
        x=40.0, y=0.0, heading=0.0, velocity=0.0,
        length=4.5, width=2.0,
        predicted_trajectory=np.array([[40.0, 0.0]] * n_points)
    )

    lane_centers = np.array([[i, 0.0] for i in range(0, 100, 2)] +
                            [[i, 3.5] for i in range(0, 100, 2)])

    route = np.array([[i, 0.0] for i in range(0, 100, 2)])

    context = SceneContext(
        agents=[obstacle],
        lane_centers=lane_centers,
        road_boundaries=None,
        speed_limit=13.89,  # 50 km/h
        ego_route=route,
    )

    return [traj1, traj2, traj3, traj4], context


def main():
    parser = argparse.ArgumentParser(description='Classical Trajectory Scorer Demo')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    args = parser.parse_args()

    if args.demo or True:  # Always run demo for now
        trajectories, context = create_demo_trajectories()
        scorer = ClassicalTrajectoryScorer()

        print("=" * 70)
        print("Classical Trajectory Scorer - Demo")
        print("=" * 70)
        print(f"\nScenario: Obstacle ahead at (40, 0). Ego at origin, speed=10 m/s")
        print(f"Speed limit: {context.speed_limit:.1f} m/s ({context.speed_limit*3.6:.0f} km/h)")
        print(f"\nCandidate trajectories:")
        print(f"  1. Lane change left (smooth lateral maneuver)")
        print(f"  2. Slow down (decelerate to avoid)")
        print(f"  3. Maintain speed (no reaction - COLLISION)")
        print(f"  4. Aggressive swerve (oscillating - uncomfortable)")
        print()

        names = ['Lane Change Left', 'Slow Down', 'Maintain Speed', 'Aggressive Swerve']

        for i, (traj, name) in enumerate(zip(trajectories, names)):
            result = scorer.score(traj, context)
            print(f"\n{'-' * 50}")
            print(f"Trajectory {i+1}: {name}")
            print(f"  Total Score: {result['total_score']:.3f}")
            print(f"  Sub-costs:")
            for cost_name, cost_val in result['sub_costs'].items():
                weight = getattr(scorer.weights, cost_name, 1.0)
                print(f"    {cost_name:20s}: {cost_val:.4f} (weight={weight:.1f})")

        # Select best
        best_idx, best_traj, best_result = scorer.select_best(trajectories, context)
        print(f"\n{'=' * 70}")
        print(f"BEST TRAJECTORY: #{best_idx+1} - {names[best_idx]}")
        print(f"Score: {best_result['total_score']:.3f}")
        print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
