from setuptools import setup

package_name = 'meu_pacote'
data_files=[]
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/robot_launch.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/Trainee_EDROM.wbt']))
data_files.append(('share/' + package_name + '/resource', ['resource/ur5e.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vitor',
    maintainer_email='vitormoraeslages@gmail.com',
    description='Pacote Trainee EDROM',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ur5e_driver = meu_pacote.ur5e_driver:main',
        ],
    },
)