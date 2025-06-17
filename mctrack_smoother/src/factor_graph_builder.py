#!/usr/bin/env python3
import gtsam
from gtsam import Pose2, BetweenFactorPose2, PriorFactorPose2, noiseModel

def build_graph_from_track(track):
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
    odom_noise  = noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2])

    for i, step in enumerate(track):
        key = gtsam.symbol('x', i)
        pose = Pose2(step['x'], step['y'], step['yaw'])
        initial.insert(key, pose)

        if i == 0:
            graph.add(PriorFactorPose2(key, pose, prior_noise))
        else:
            prev = track[i - 1]
            delta = Pose2(step['x'] - prev['x'], step['y'] - prev['y'], step['yaw'] - prev['yaw'])
            graph.add(BetweenFactorPose2(gtsam.symbol('x', i - 1), key, delta, odom_noise))

    return graph, initial

