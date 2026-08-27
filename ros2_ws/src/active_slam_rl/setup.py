from setuptools import find_packages, setup


package_name = 'active_slam_rl'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(
        exclude=['test']
    ),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Mohammad Owais',
    maintainer_email=(
        'owais03321-glitch@users.noreply.github.com'
    ),
    description=(
        'ROS 2 Active SLAM research prototype using '
        'frontier exploration, Nav2, and MaskablePPO '
        'for learned frontier selection.'
    ),
    license='Apache-2.0',
    url=(
        'https://github.com/'
        'owais03321-glitch/active-slam-rl'
    ),
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            (
                'map_monitor = '
                'active_slam_rl.map_monitor:main'
            ),
            (
                'frontier_detector = '
                'active_slam_rl.frontier_detector:main'
            ),
            (
                'nav2_goal_sender = '
                'active_slam_rl.nav2_goal_sender:main'
            ),
            (
                'frontier_explorer = '
                'active_slam_rl.frontier_explorer:main'
            ),
            (
                'rl_observation_node = '
                'active_slam_rl.rl_observation_node:main'
            ),
            (
                'baseline_metrics = '
                'active_slam_rl.baseline_metrics:main'
            ),
        ],
    },
)
