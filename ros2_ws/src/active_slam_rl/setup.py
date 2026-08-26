from setuptools import find_packages, setup

package_name = 'active_slam_rl'

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
    maintainer='mohammad-owais',
    maintainer_email='mohammad-owais@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
	    'map_monitor = active_slam_rl.map_monitor:main',
	    'frontier_detector = active_slam_rl.frontier_detector:main',
	    'nav2_goal_sender = active_slam_rl.nav2_goal_sender:main',
	    'frontier_explorer = active_slam_rl.frontier_explorer:main',
        ],
    },
)
