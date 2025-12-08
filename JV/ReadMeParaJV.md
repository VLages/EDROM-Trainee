cd ~/ws_edrom
colcon build
source install/setup.bash
ros2 run ur5e_sim coords_subscriber

webots ~/webots_projects/ur5e_sim/worlds/arm_world.wbt
