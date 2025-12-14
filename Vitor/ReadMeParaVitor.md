Para inicar o programa, colo a pasta "meu_pacote" dentro do diretório "/ros2_ws/src".
Abra o terminal e coloque esses códigos:
  cd ~/ros2_ws/src
  colcon build --symlink-install --packages-select meu_pacote
  source install/setup.bash
  ros2 launch meu_pacote robot_launch.py
Depois do mundo ser processado, abra um novo terminal e inice o arqui "ur5e_driver.py" através do comando:
  python3 ur5e_driver.py
