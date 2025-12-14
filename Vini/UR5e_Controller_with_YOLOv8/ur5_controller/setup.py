from setuptools import find_packages, setup

package_name = 'ur5_controller'

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
    maintainer='vital',
    maintainer_email='vital@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
      entry_points={
        'console_scripts': [
            'ur5_driver = ur5_controller.my_robot_driver:main',
        ],
    },
)
