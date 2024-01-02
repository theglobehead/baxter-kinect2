import ctypes
import json
import math
from math import atan2, pi

import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

class Kinect():
    """ Offers some more functionality and access via the
        PyKinect2 interface """
    def __init__(self):
        # From the kinect we only need the depth frame
        kinect_frame_types = PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
        self._kinect = PyKinectRuntime.PyKinectRuntime(kinect_frame_types)

        self.color_frame = None
        self.depth_frame = None
        self.depth_frame_original = None

        self.RGB_WIDTH = self._kinect.color_frame_desc.Width
        self.RGB_HEIGHT = self._kinect.color_frame_desc.Height
        self.DEPTH_WIDTH = self._kinect.depth_frame_desc.Width
        self.DEPTH_HEIGHT = self._kinect.depth_frame_desc.Height

        self.has_new_color_frame = self._kinect.has_new_color_frame
        self.get_last_color_frame = self._kinect.get_last_color_frame
        self.has_new_depth_frame = self._kinect.has_new_depth_frame
        self.get_last_depth_frame = self._kinect.get_last_depth_frame

        # create C-style Type
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * (self.DEPTH_WIDTH * self.DEPTH_HEIGHT)
        # create a buffer to reuse when mapping to CameraSpacePoint
        self.depth_to_csp_buffer = TYPE_CameraSpacePoint_Array()

    def close(self):
        self._kinect.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def isolate_hsv_color_range(image, lower_vals: tuple, upper_vals: tuple):
        image = cv2.blur(image, (8, 8))
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, lower_vals, upper_vals)

        return mask

    @staticmethod
    def get_pixel_average_cords(mask) -> tuple:
        result = None

        try:
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            big_contour = max(contours, key=cv2.contourArea)

            M = cv2.moments(big_contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

            result = centroid_x, centroid_y
        except Exception:
            pass

        return result

    def map_depth_frame_to_camera_space_points(self, d_frame):
        # numpy.ndarray(424*512) -> numpy.ndarray(424x512x3)
        """ From a depth-frame (numpy.ndarray) it calculates the
            coordinates in cameraspace. The returned array
            holds 3 floats for each coordinate.
        """
        L = self.DEPTH_WIDTH * self.DEPTH_HEIGHT

        # size must be what the kinect offers as depth frame
        assert L == d_frame.size
        # cstyle pointer to our 1d depth_frame data
        ptr_depth = np.ctypeslib.as_ctypes(d_frame.flatten())
        # calculate cameraspace coordinates
        error_state = self._kinect._mapper.MapDepthFrameToCameraSpace(
                                                                L, ptr_depth,
                                                                L, self.depth_to_csp_buffer)
        # 0 means everything's ok, otherwise failed!
        if error_state:
            raise "Could not map depth frame to camera space! " + str(error_state)

        # convert back from ctype to numpy.ndarray
        pf_csps = ctypes.cast(self.depth_to_csp_buffer, ctypes.POINTER(ctypes.c_float))
        data = np.ctypeslib.as_array(pf_csps, shape=(self.DEPTH_HEIGHT, self.DEPTH_WIDTH,
                                                     3))
        del pf_csps, ptr_depth, d_frame
        return np.copy(data)

    def adjust_color_cords_to_depth_cords(self, x: int, y: int):
        x_offset = x - self.RGB_WIDTH / 2
        y_offset = y - self.RGB_HEIGHT / 2

        x_cof = self.DEPTH_WIDTH / self.RGB_WIDTH*1.25
        y_cof = self.DEPTH_HEIGHT / self.RGB_HEIGHT*0.9

        depth_x = int(self.DEPTH_WIDTH/2 + x_offset * x_cof) - 10
        depth_y = int(self.DEPTH_HEIGHT/2 + y_offset * y_cof)

        return depth_x, depth_y

    def get_location_of_joint(self, x: int, y: int, csp_frame):
        max_z_data = None

        try:
            csp_data = csp_frame[y:y+2, x:x+2]
            csp_data = np.reshape(csp_data, (-1, csp_data.shape[-1]))

            max_z_data_idx = np.argmax(csp_data[:, 2])
            max_z_data = csp_data[max_z_data_idx]
        except Exception:
            pass

        return max_z_data

    def update_joint_mean_cords(self, joint):
        joint.pixel_mask = self.isolate_hsv_color_range(
            color_frame,
            joint.lower_color_values,
            joint.upper_color_values,
        )

        if joint.color_name == "yellow":
            joint.pixel_mask = joint.pixel_mask[0:self.RGB_HEIGHT - 100, 0:self.RGB_WIDTH-500]
        elif joint.color_name == "purple":
            joint.pixel_mask = joint.pixel_mask[0:self.RGB_HEIGHT, 0:self.RGB_WIDTH]

        new_mean_cords = self.get_pixel_average_cords(joint.pixel_mask)
        joint.mean_cords = new_mean_cords if new_mean_cords is not None else joint.mean_cords


class Baxter:
    def __init__(self):
        self.chest = Joint(
            color_name="purple",
        )
        self.shoulder = Joint(
            color_name="purple"
        )
        self.elbow = Joint(
            color_name="yellow",
            approximate_radius_m=0.07,
        )
        self.wrist_1 = Joint(
            color_name="blue",
            approximate_radius_m=0.06,
        )
        self.wrist_2 = Joint(
            color_name="green",
            approximate_radius_m=0.035,
        )
        self.moving_joints = [
            self.elbow,
            self.wrist_1,
            self.wrist_2,
        ]

        self.upper_arm = Limb(
            start_joint=self.shoulder,
            end_joint=self.elbow,
            length_m=0.4,
        )
        self.forearm = Limb(
            start_joint=self.elbow,
            end_joint=self.wrist_1,
            length_m=0.4,
        )
        self.wrist = Limb(
            start_joint=self.wrist_1,
            end_joint=self.wrist_2,
            length_m=0.4,
        )
        self.limbs = [
            self.upper_arm,
            self.forearm,
            self.wrist,
        ]

        self.s0 = 0
        self.s1 = 0
        self.e0 = 0
        self.e1 = 0
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0

    def calculate_ik(self, ee_x, ee_y, ee_z):
        """
        We are using the trigonometric method. There is a triangle abc
        ab is the upper ar
        bc the lower arm
        ac the distance between the shoulder and EE
        """
        angles = {
            'right_s0': 0,
            'right_s1': 0,
            'right_e0': 0,
            'right_e1': 0,
            'right_w0': 0,
            'right_w1': 0,
            'right_w2': 0
        }

        s0_z = self.shoulder.z - ee_z
        s0_x = self.shoulder.x - ee_x
        s0 = atan2(s0_x, s0_z) + 0.785  # 0.785 radians is 45 degrees
        if s0 and not math.isnan(s0):
            angles["right_s0"] = s0

        ac = [
            self.shoulder.x - ee_x,
            self.shoulder.y - ee_y,
            self.shoulder.z - ee_z,
        ]
        ac = math.sqrt(ac[0] ** 2 + ac[1] ** 2 + ac[2] ** 2)
        ab = 0.4
        bc = 0.65

        e1 = 3.14 - self.calculate_joint_angle(ab, bc, ac)
        alpha = self.calculate_joint_angle(ab, ac, bc)

        alpha_prim = math.atan(
            (ee_y - self.shoulder.y) / math.sqrt((ee_z - self.shoulder.z)**2 + (ee_x - self.shoulder.x)**2)
        )

        print(alpha, alpha_prim)
        s1 = (alpha - alpha_prim) * -1

        angles["right_e1"] = e1
        angles["right_s1"] = s1

        return angles

    @ staticmethod
    def calculate_joint_angle(l1, l2, dist):
        angle_c = 0

        try:
            cos_c = (l1 ** 2 + l2 ** 2 - dist ** 2) / (2 * l1 * l2)
            angle_c = math.acos(cos_c)
            # angle_c_degrees = math.degrees(angle_c)
        except ValueError:
            pass

        return angle_c

    def calculate_triangle_angles(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):
        # Calculate vectors AB, BC, and CA
        ab = [b1 - a1, b2 - a2, b3 - a3]
        bc = [c1 - b1, c2 - b2, c3 - b3]
        ca = [a1 - c1, a2 - c2, a3 - c3]

        # # Calculate magnitudes of vectors AB, BC, and CA
        mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)
        mag_ca = math.sqrt(ca[0] ** 2 + ca[1] ** 2 + ca[2] ** 2)

        return self.calculate_joint_angle(mag_ab, mag_bc, mag_ca, )

    def calculate_actuator_angles(self):
        # Calculate S0
        s0_s1_z = self.shoulder.z - self.elbow.z
        s0_s1_x = self.shoulder.x - self.elbow.x
        s0 = atan2(s0_s1_x, s0_s1_z)
        if s0 and not math.isnan(s0):
            self.s0 = s0 + 0.785  # 0.785 radians is 45 degrees

        # print(self.shoulder.x, self.shoulder.y, self.shoulder.z)
        # print(self.elbow.x, self.elbow.y, self.elbow.z)
        # print(self.wrist_1.x, self.wrist_1.y, self.wrist_1.z)
        # print(self.wrist_2.x, self.wrist_2.y, self.wrist_2.z)

        s1 = self.calculate_triangle_angles(
            self.shoulder.x, self.shoulder.y - 50, self.shoulder.z,
            self.shoulder.x, self.shoulder.y, self.shoulder.z,
            self.elbow.x, self.elbow.y, self.elbow.z,
        )
        if not math.isnan(s1):
            self.s1 = s1 - 1.57  # 1.57 radians is 90 degrees

        e1 = self.calculate_triangle_angles(
            self.shoulder.x, self.shoulder.y, self.shoulder.z,
            self.elbow.x, self.elbow.y, self.elbow.z,
            self.wrist_1.x, self.wrist_1.y, self.wrist_1.z,
        )
        if not math.isnan(e1):
            self.e1 = abs(e1 - 3.14)

        w1 = self.calculate_triangle_angles(
            self.elbow.x, self.elbow.y, self.elbow.z,
            self.wrist_1.x, self.wrist_1.y, self.wrist_1.z,
            self.wrist_2.x, self.wrist_2.y, self.wrist_2.z,
        )
        if not math.isnan(w1):
            self.w1 = abs(w1 - 3.14)

        print("-----------------------------------------")
        print("self.s0:", math.degrees(self.s0))
        print("self.s1:", math.degrees(self.s1))
        print("self.e1:", math.degrees(self.e1))
        print("self.w1:", math.degrees(self.w1))
        print("-----------------------------------------")


class Joint:
    color_values = {
        "yellow": ((30, 70, 140), (35, 255, 255)),
        "blue": ((100, 120, 20), (150, 255, 255)),
        "green": ((45, 80, 105), (91, 255, 255)),
        "purple": ((105, 40, 120), (150, 100, 160)),
        "orange": ((10, 110, 160), (30, 255, 255)),
    }

    def __init__(self, color_name: str, approximate_radius_m: float = 0):
        self.x = 0
        self.y = 0
        self.z = 0

        self.approximate_radius_m = approximate_radius_m

        self.mean_cords = (0, 0)

        self.pixel_mask = None
        self.color_name = color_name
        self.lower_color_values = Joint.color_values[color_name][0]
        self.upper_color_values = Joint.color_values[color_name][1]


class Limb:
    def __init__(self, start_joint: Joint, end_joint: Joint, length_m: float):
        self.start_joint = start_joint
        self.end_joint = end_joint
        self.length_m = length_m

cv2.namedWindow("RGB stream")
cv2.namedWindow("Depth stream")
cv2.namedWindow("Debug")

kinect = Kinect()
baxter = Baxter()
ee = Joint(color_name="orange")

running = True
while running:
    has_new_color_frame = kinect.has_new_color_frame()
    has_new_depth_frame = kinect.has_new_depth_frame()

    if has_new_color_frame:
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((kinect.RGB_HEIGHT, kinect.RGB_WIDTH, 4))
        color_frame = cv2.convertScaleAbs(color_frame)

        # Update x, y cords in color frame
        if not any(baxter.chest.mean_cords):
            kinect.update_joint_mean_cords(baxter.chest)
        elif any((baxter.shoulder.x, baxter.shoulder.y, baxter.shoulder.z)):
            baxter.shoulder.mean_cords = (
                baxter.chest.mean_cords[0] + 180,
                baxter.chest.mean_cords[1],
            )
        for joint in baxter.moving_joints:
            kinect.update_joint_mean_cords(joint)

        # For finding the end effector
        kinect.update_joint_mean_cords(ee)
        cv2.circle(
            color_frame, ee.mean_cords, radius=5, color=(0, 0, 255), thickness=-1
        )

        # Visualize the limbs and joints
        for limb in baxter.limbs:
            if any(limb.start_joint.mean_cords):
                cv2.circle(
                    color_frame, limb.start_joint.mean_cords, radius=10, color=(0, 0, 255), thickness=-1
                )
            if any(limb.end_joint.mean_cords):
                cv2.circle(
                    color_frame, limb.end_joint.mean_cords, radius=10, color=(0, 0, 255), thickness=-1
                )

            if any(limb.start_joint.mean_cords) and any(limb.end_joint.mean_cords):
                cv2.line(color_frame, limb.end_joint.mean_cords, limb.start_joint.mean_cords, color=(0, 0, 255), thickness=3)

        debug_mask = baxter.wrist_2.pixel_mask
        debug_frame = cv2.bitwise_and(debug_mask, debug_mask, mask=debug_mask)
        debug_frame = cv2.resize(debug_frame, (int(kinect.RGB_WIDTH/2), int(kinect.RGB_HEIGHT/2)))

        kinect.color_frame = color_frame

        color_frame = cv2.resize(color_frame, (int(kinect.RGB_WIDTH/2), int(kinect.RGB_HEIGHT/2)))
        cv2.imshow("RGB stream", color_frame)
        cv2.imshow("Debug", debug_frame)

    if has_new_depth_frame:
        depth_frame = kinect.get_last_depth_frame()
        kinect.depth_frame_original = depth_frame
        depth_frame = depth_frame.reshape((kinect.DEPTH_HEIGHT, kinect.DEPTH_WIDTH))
        depth_frame = cv2.convertScaleAbs(depth_frame, alpha=0.05)

        for limb in baxter.limbs:
            s_adjusted_mean_cords = None
            e_adjusted_mean_cords = None
            if any(limb.start_joint.mean_cords):
                s_adjusted_mean_cords = kinect.adjust_color_cords_to_depth_cords(
                    limb.start_joint.mean_cords[0],
                    limb.start_joint.mean_cords[1]
                )

                cv2.circle(
                    depth_frame, s_adjusted_mean_cords, radius=10, color=(0, 0, 255), thickness=-1
                )
            if any(limb.end_joint.mean_cords):
                e_adjusted_mean_cords = kinect.adjust_color_cords_to_depth_cords(
                    limb.end_joint.mean_cords[0],
                    limb.end_joint.mean_cords[1]
                )

                cv2.circle(
                    depth_frame, e_adjusted_mean_cords, radius=10, color=(0, 0, 255), thickness=-1
                )

            if s_adjusted_mean_cords and e_adjusted_mean_cords:
                cv2.line(depth_frame, s_adjusted_mean_cords, e_adjusted_mean_cords, color=(0, 0, 255), thickness=3)

        cv2.imshow("Depth stream", depth_frame)
        kinect.depth_frame = depth_frame

    if has_new_depth_frame and has_new_color_frame:
        csp_frame = kinect.map_depth_frame_to_camera_space_points(kinect.depth_frame_original)

        if any(baxter.chest.mean_cords) and not any((baxter.chest.x, baxter.chest.y, baxter.chest.z)):
            x, y = kinect.adjust_color_cords_to_depth_cords(
                baxter.chest.mean_cords[0],
                baxter.chest.mean_cords[1]
            )
            location = kinect.get_location_of_joint(x, y, csp_frame)
            if location is not None:
                baxter.chest.x, baxter.chest.y, baxter.chest.z = location
                baxter.chest.z = baxter.chest.z - baxter.chest.approximate_radius_m
            baxter.shoulder.x, baxter.shoulder.y, baxter.shoulder.z = baxter.chest.x + 0.28, baxter.chest.y, baxter.chest.z

        for joint in baxter.moving_joints:
            x, y = kinect.adjust_color_cords_to_depth_cords(joint.mean_cords[0], joint.mean_cords[1])
            location = kinect.get_location_of_joint(x, y, csp_frame)
            if location is not None:
                joint.x, joint.y, joint.z = location
                joint.z = joint.z - joint.approximate_radius_m

        baxter.calculate_actuator_angles()

        # For EE
        x, y = kinect.adjust_color_cords_to_depth_cords(ee.mean_cords[0], ee.mean_cords[1])
        location = kinect.get_location_of_joint(x, y, csp_frame)
        if location is not None:
            ee.x, ee.y, ee.z = location
            ee.z = ee.z - ee.approximate_radius_m

        goal_angles = baxter.calculate_ik(ee.x, ee.y, ee.z)

        # Used by the server to send data to baxter
        with open("angle_data.txt", 'w') as file:
            file.write(json.dumps({
                "current_angles": {
                    "right_s0": baxter.s0,
                    "right_s1": baxter.s1 * -1,
                    "right_e1": baxter.e1,
                    "right_w1": baxter.w1,
                },
                "goal_angles": goal_angles,
                # "goal_angles": {
                #     'right_s0': 0,
                #     'right_s1': 0,
                #     'right_e0': 0,
                #     'right_e1': 0,
                #     'right_w0': 0,
                #     'right_w1': 0,
                #     'right_w2': 0
                # },
            }))

    key = cv2.waitKey(1)
    if key == 27:
        running = False

kinect.close()
cv2.destroyAllWindows()
