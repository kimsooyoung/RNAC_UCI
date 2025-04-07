import mujoco
import mujoco_viewer

ROBOT_SCENE = "./unitree_a1/scene.xml" # Robot scene
# ROBOT_SCENE = "./unitree_go1/scene_position.xml" # Robot scene
# ROBOT_SCENE = "./unitree_go1/scene_torque.xml" # Robot scene

model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
data = mujoco.MjData(model)

viewer = mujoco_viewer.MujocoViewer(model, data)
mujoco.mj_resetDataKeyframe(model, data, 0)

# step at least once to load model in viewer
mujoco.mj_step(model, data)  

for i in range(1000):
    while viewer.is_alive:
        mujoco.mj_step(model, data)
        mujoco.mj_rnePostConstraint(model, data)
        mujoco.mj_fwdActuation(model, data)

        data.ctrl[:] = model.key_ctrl

        viewer.render()
    else:
        break
viewer.close()
