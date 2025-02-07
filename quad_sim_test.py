import mujoco
import mujoco_viewer

ROBOT_SCENE = "./unitree_a1/scene.xml" # Robot scene

mj_model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data)

# step at least once to load model in viewer
mujoco.mj_step(mj_model, mj_data)  

while viewer.is_alive:
    viewer.render()
viewer.close()
