from setuptools import find_packages, setup

package_name = 'ur5e_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jvsc152',
    maintainer_email='jvsc152@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'coords_subscriber = ur5e_sim.coords_subscriber:main',
            'follow node = ur5e_sim.follow_node:main',
            'pick_place = ur5e_sim.pick_place:main',
        ],
    },
)
