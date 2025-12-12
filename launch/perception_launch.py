from math import inf
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os,json,yaml,subprocess,platform

def update_calibration_params(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    for camera in config['cameras']:
        topic = camera['topic']['camera_info']
        
        cmd = f"bash -c 'source /opt/ros/humble/setup.bash && ros2 topic echo {topic} --once'"
        # if(platform.machine() == "x86_64"):
        #     timeout = 5
        # else:
        #     timeout = 10
        timeout = 10
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        info = yaml.safe_load(result.stdout.strip()[:-4])

        camera["calibration"]["fx"] = info["k"][0]
        camera["calibration"]["fy"] = info["k"][4]
        camera["calibration"]["cx"] = info["k"][2]
        camera["calibration"]["cy"] = info["k"][5]

        print(camera["calibration"])

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=False)

def declare_configurable_parameters(parameters):
    return [
        DeclareLaunchArgument(param["name"], default_value=param["default_value"])
        for param in parameters
    ]
    
def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])


def generate_launch_description():

    # print(get_package_share_directory('perception'))

    root = get_package_share_directory('perception')

    if(os.path.exists(os.path.join(root, 'config', 'camera.json'))):
        update_calibration_params(os.path.join(root, 'config', 'camera.json'))

    node_params = [
        {"name": "node_name", "default_value": "perception_node"},
        {"name": "log_level", "default_value": "info"},

        {"name": "root", "default_value": root + "/"},
        {"name": "camera_config_path", "default_value": os.path.join(root, 'config', 'camera.json')},
        {"name": "laser_config_path", "default_value": os.path.join(root, 'config', 'laser.json')},
        {"name": "client_config_path", "default_value": os.path.join(root, 'config', 'client.json')},
        {"name": "model_config_path", "default_value": os.path.join(root, 'config', 'model.json')},

        {"name": "ms", "default_value": "False"}, # ms level precision

        {"name": "infer", "default_value": "False"},
        {"name": "pub_laser", "default_value": "False"},
        {"name": "pub_pc", "default_value": "False"},

        {"name": "show", "default_value": "False"},

        {"name": "save", "default_value": "False"},
        {"name": "save_dir", "default_value": "/home/sunrise/Desktop/dataset"},
        {"name": "save_name", "default_value": "12-12"}
    ]

    launch = declare_configurable_parameters(node_params)

    launch.append(Node(
        package='perception',
        executable='inference',
        name=LaunchConfiguration('node_name'),
        output='screen',
        parameters=[set_configurable_parameters(node_params)],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    ))

    return LaunchDescription(launch)
