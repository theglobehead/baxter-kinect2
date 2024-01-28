import math
import rospy
import baxter_interface

import json
import http.client

import signal


# Initialize the node
rospy.init_node('limb_simple_move', "rsdk_ik_service_client")

# Connect to the right limb
limb = baxter_interface.Limb('right')

kinect_url = "192.168.1.153:5000"
iterations = 35

joint_step_size = {
    'right_s0': 0.1,
    'right_s1': 0.1,
    'right_e0': 0,
    'right_e1': 0.1,
    'right_w0': 0,
    'right_w1': 0.1,
    'right_w2': 0,
}


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")


def get_data_from_url(url):
    try:
        # Split the url into hostname and port
        parts = url.split(":")
        hostname = parts[0]
        port = int(parts[1])

        connection = http.client.HTTPConnection(hostname, port)
        connection.request("GET", "/")
        response = connection.getresponse()

        if response.status == 200:
            json_data = json.loads(response.read().decode())
            return json_data
        else:
            print("Error making request to '{}': {} {}".format(url, response.status, response.reason))
        connection.close()
    except Exception as e:
        print("Error making request to '{}': {}".format(url, str(e)))


def main():
    # Start joint positions are 0 positions
    start_joint_positions = {
        'right_s0': 0,
        'right_s1': 0,
        'right_e0': 0,
        'right_e1': 0,
        'right_w0': 0,
        'right_w1': 0,
        'right_w2': 0,
    }

    for _ in range(5):
        limb.move_to_joint_positions(start_joint_positions)

    # end_joint_positions = {
    #     'right_s0': 0,
    #     'right_s1': 0,
    #     'right_e0': 0,
    #     'right_e1': 0,
    #     'right_w0': 0,
    #     'right_w1': 0,
    #     'right_w2': 0,
    # }

    end_joint_positions = None

    for i in range(iterations):
        print("INFO: iteration nr.", i + 1)

        kinect_data = get_data_from_url(kinect_url)
        if not kinect_data:
            kinect_data = get_data_from_url(kinect_url)  # Retry

        if end_joint_positions is None:
            end_joint_positions = kinect_data["goal_angles"]
            print("INFO: End joint possitions calculated")

        baxter_angles = limb.joint_angles()
        kinect_angles = kinect_data["current_angles"]
        kinect_angles["right_e0"] = 0
        kinect_angles["right_w0"] = 0
        kinect_angles["right_w2"] = 0

        current_target_positions = baxter_angles.copy() # Make a copy to avoid modifying the original dict
        for joint in current_target_positions:
            # Changes the direction of the steps if needed
            if ((kinect_angles[joint] > end_joint_positions[joint] and joint_step_size[joint] > 0) or
                    (kinect_angles[joint] < end_joint_positions[joint] and joint_step_size[joint] < 0)
            ):
                joint_step_size[joint] *= -1

            # If step too big, makes it smaller
            if math.fabs(joint_step_size[joint]) > math.fabs(kinect_angles[joint] - end_joint_positions[joint]):
                    joint_step_size[joint] *= 0.7

            # If step is too small for baxter to move accurately, consider the joint solved
            if math.fabs(joint_step_size[joint]) < 0.1 / 2**4 and joint_step_size[joint] != 0:
                    joint_step_size[joint] = 0

            # Make robot move one step
            current_target_positions[joint] += joint_step_size[joint]

        current_target_positions["right_e0"] = 0
        current_target_positions["right_w0"] = 0
        current_target_positions["right_w2"] = 0

        # Check if endpoint achieved
        joint_step_size_sum = 0
        for joint in joint_step_size:
            joint_step_size_sum += math.fabs(joint_step_size[joint])
        if not joint_step_size_sum:
            print("INFO: Endpoint achieved!")
            break

        # For easier debugging
        for joint in kinect_angles:
            print(joint, "| current_kinect:", kinect_angles[joint], "| baxter_angle:", baxter_angles[joint], "| current_target:", current_target_positions[joint], "| end_goal:", end_joint_positions[joint], "| step:", joint_step_size[joint])

        for _ in range(5):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            try:
                limb.move_to_joint_positions(current_target_positions)

                signal.alarm(0)
            except TimeoutError:
                print("Function timed out")

main()
print("done")