# Genesis
# track_lin_reward= 0.923779308795929
# track_ang_reward= 0.9902310967445374
# penalize_z= 0.0006619965424761176
# action_rate_penalty= 17.252445220947266
# pose_deviation_penalty= 1.6526377201080322
# height_deviation_penalty= 2.50177072302904e-06

# Mujoco
track_lin_reward= 0.9768024841314993
track_ang_reward= 0.04995897079138872
penalize_z= 0.007446677018903214
action_rate_penalty= 7.445662070913276
pose_deviation_penalty= 10.429578789706687
height_deviation_penalty= 0.0007688640673757057
rew_contact_forces= 0.0

# Total reward
total_reward = (
    1.0 * track_lin_reward +
    0.2 * track_ang_reward +
    -1.0 * penalize_z +
    -0.005 * action_rate_penalty +
    -0.1 * pose_deviation_penalty +
    -50.0 * height_deviation_penalty
)

print(f"total_reward: {total_reward}")