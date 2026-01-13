from ast import arg
from math import e
import argparse,subprocess

import os,json,yaml

ros_build = "humble"

def update_shape(topic):
    cmd = "source /opt/ros/{ros_build}/setup.bash && ros2 topic echo {topic} --once --no-arr".format(ros_build=ros_build, topic=topic)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=10
        )
        result = result.stdout.strip()[:-4]
        for index,line in enumerate(result.split('\n')):
            if line.startswith("header"):
                result = "\n".join(result.split('\n')[index:])
                break
        result = yaml.safe_load(result)
        shape = [result["width"], result["height"]]
        return shape
    except Exception as e:
        print(f" please make sure the topic {topic} is exist")
        return None

def update_calib(topic):
    cmd = "source /opt/ros/{ros_build}/setup.bash && ros2 topic echo {topic} --once".format(ros_build=ros_build, topic=topic)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=5
        )
        result = result.stdout.strip()[:-4]
        for index,line in enumerate(result.split('\n')):
            if line.startswith("header"):
                result = "\n".join(result.split('\n')[index:])
                break
        result = yaml.safe_load(result)
        calib = {
            "fx": result["k"][0],
            "fy": result["k"][4],
            "cx": result["k"][2],
            "cy": result["k"][5],
            "baseline": 0.06997
        }
        return calib
    except Exception as e:
        print(f" please make sure the topic {topic} is exist")
        return None


def auto_update_shape(args):
    camera_config_path = args.file
    with open(camera_config_path, 'r') as f:
        camera_json = json.load(f)

    camera_json['camera']['shape'] = update_shape(camera_json['camera']['topic']['image1'])
    camera_json['camera']['calib'] = update_calib(camera_json['camera']['topic']['camera_info'])

    with open(camera_config_path, 'w') as f:
        json.dump(camera_json, f, indent=4, sort_keys=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='', help='store config path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    auto_update_shape(args)