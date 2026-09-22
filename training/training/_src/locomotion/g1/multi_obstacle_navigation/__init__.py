"""Multi-obstacle DPCBF navigation for the Unitree G1."""

from training._src.locomotion.g1.multi_obstacle_navigation.navigation import (
    MultiObstacleNavigation,
    default_config,
)

__all__ = ["MultiObstacleNavigation", "default_config"]
