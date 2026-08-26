"""Friday's spatial understanding: turning LiDAR sweeps into a map.

Pure Python on numpy/scipy, which are already in the venv. No ROS, no new
system packages, no new services -- see CLAUDE.md Part 1 for why that is not
a stylistic preference.

    core     points, the likelihood field, and the scan matcher
    grid     the occupancy map itself, in log-odds

Nothing here talks to hardware. Scans come in as {bearing: mm} dicts exactly
as sensors.Lidar.scan() returns them, so every part of this is testable
against recorded sweeps with no robot attached.
"""

from slam.core import (LikelihoodField, match_confidence, match_pose,
                       match_translation, points_from_scan, transform)
from slam.grid import OccupancyGrid
from slam.transient import TransientLayer

__all__ = ["LikelihoodField", "OccupancyGrid", "TransientLayer",
           "match_confidence", "match_pose", "match_translation",
           "points_from_scan", "transform"]
