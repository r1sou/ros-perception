from math import inf
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os

def declare_configurable_parameters(parameters):
    return [
        DeclareLaunchArgument(param["name"], default_value=param["default_value"])
        for param in parameters
    ]
    
def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])


def generate_launch_description():

    root = get_package_share_directory('perception')

    node_params = [
        {"name": "node_name", "default_value": "perception_node"},
        {"name": "log_level", "default_value": "info"},

        {"name": "project_root", "default_value": root},
        {"name": "client_config_path", "default_value": 'client.json'},
        {"name": "camera_config_path", "default_value": 'camera.json'},
        {"name": "model_config_path", "default_value": 'model.json'},
        {"name": "laser_config_path", "default_value": 'laser.json'},
        {"name": "record_config_path", "default_value": 'record.json'},

        {"name": "record", "default_value": "False"},
        {"name": "followme", "default_value": "False"},

        {"name": "show", "default_value": "False"},
        {"name": "debug", "default_value": "False"}
    ]

    launch = declare_configurable_parameters(node_params)

    launch.append(Node(
        package='perception',
        executable='perception_node',
        name=LaunchConfiguration('node_name'),
        output='screen',
        parameters=[set_configurable_parameters(node_params)],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    ))

    return LaunchDescription(launch)