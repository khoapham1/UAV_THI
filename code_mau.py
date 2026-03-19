#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
######## IMPORTS #########
import rospy
from sensor_msgs.msg import Image
import cv2
import time
import math
import numpy as np
import ros_numpy as rnp
from dronekit import connect, VehicleMode
from pymavlink import mavutil
# from lib_uav.lib_color import ColorSegmenter, CenterTracker
# from lib_uav.lib_pid import PIDController
######## VARIABLES ########
vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)
vehicle.parameters['PLND_ENABLED'] = 1
vehicle.parameters['PLND_TYPE'] = 1
vehicle.parameters['PLND_EST_TYPE'] = 0
vehicle.parameters['LAND_SPEED'] = 30  # cm/s

velocity = -0.5  # m/s
takeoff_height = 4  # m

horizontal_res = 800
vertical_res = 600

horizontal_fov = 62.2 * math.pi / 180
vertical_fov = 48.8 * math.pi / 180

newimg_pub = rospy.Publisher('/camera/color/image_new', Image, queue_size=10)

time_last = 0
time_to_wait = 0.1
found_count = 0
notfound_count = 0

# State machine step
mission_step = 0
hold_start_time = 0
in_target_zone = False

# Threshold parameters for H detection
MIN_CONTOUR_AREA = 1000       # Minimum contour area to consider
ASPECT_RATIO_MIN = 0.8        # Minimum width/height ratio
ASPECT_RATIO_MAX = 1.2        # Maximum width/height ratio
SOLIDITY_THRESH = 0.7         # Minimum solidity (contour area / convex hull area)
HOLE_RATIO_MIN = 0.1          # Minimum hole area ratio
HOLE_RATIO_MAX = 0.5          # Maximum hole area ratio

#### PID parameters
class PIDController(object):
    def __init__(self, Kp, Ki, Kd, max_output):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output
        self.reset()

    def reset(self):
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0

        self.integral += error * dt
        derivative = (error - self.last_error) / dt

        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        output = np.clip(output, -self.max_output, self.max_output)

        self.last_error = error
        self.last_time = current_time
        return output

PID_X = PIDController(Kp=0.002, Ki=0.0007, Kd=0.0, max_output=0.5) # m/s
PID_Y = PIDController(Kp=0.002, Ki=0.0007, Kd=0.0 , max_output=0.5) # m/s

######## FUNCTIONS ########
def arm_and_takeoff(targetHeight):
    print('Vehicle is now armable')
    vehicle.mode = VehicleMode('GUIDED')
    while vehicle.mode.name != 'GUIDED':
        print('Waiting for GUIDED mode')
        time.sleep(1)
    print('Vehicle in GUIDED mode')

    vehicle.armed = True
    while not vehicle.armed:
        print('Waiting for arming...')
        time.sleep(1)
    print('Armed!')

    vehicle.simple_takeoff(targetHeight)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print('Altitude: %d m' % alt)
        if alt >= 0.95 * targetHeight:
            break
        time.sleep(1)
    print('Target altitude reached!')

def send_local_ned_velocity(vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0, vx, vy, vz,
        0, 0, 0, 0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def send_land_message(x, y):
    msg = vehicle.message_factory.landing_target_encode(
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        x, y,
        0, 0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

def is_in_target_zone(error_x, error_y, threshold=35):
    return abs(error_x) < threshold and abs(error_y) < threshold

def move_with_timer(direction, duration, speed=0.5):
    vx, vy = 0, 0
    if direction == 'forward':
        vx = speed
    elif direction == 'backward':
        vx = -speed
    elif direction == 'left':
        vy = -speed
    elif direction == 'right':
        vy = speed
    else:
        return
    start_time = time.time()
    while time.time() - start_time < duration:
        send_local_ned_velocity(vx, vy, 0)
        time.sleep(0.1)
    send_local_ned_velocity(0, 0, 0)

######################
class ColorSegmenter(object):
    def __init__(self):
        self.color_ranges = {
            "blue": (np.array([100,80,80]), np.array([130,255,255])),
            "red": (np.array([0,80,80]), np.array([10,255,255])),
            "yellow": (np.array([15,80,80]), np.array([35,255,255])),
            "white": (np.array([0,0,200]), np.array([180,30,255]))  # HSV white range
        }

    def segment_color(self, img, color):
        if color not in self.color_ranges:
            raise ValueError("color {} not defined.".format(color))
        lower, upper = self.color_ranges[color]
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, lower, upper)
        binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        result = cv2.bitwise_and(img, img, mask=binary_mask)
        return result, binary_mask

    def detect_objects(self, img, color):
        _, binary_mask = self.segment_color(img, color)
        kernel = np.ones((10,10), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        _,contours, _ = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detection = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                detection.append((x, y, w, h))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 100), 2)
                cv2.circle(img, (x + w //2, y + h // 2), 35, (255, 0, 0), -1)
                cv2.putText(img, color, (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img, detection

    def segment_all(self, img):
        return {color: self.segment_color(img, color)[0] for color in self.color_ranges}

######################
class CenterTracker(object):
    global offset, circle_detect
    def __init__(self):
        self.offset = [0,130]
        self.circle_detect = 250

    def draw_center_track(self, frame):
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.arrowedLine(frame, (center_x + self.offset[0], center_y + self.offset[1]), (center_x + self.offset[0], center_y + self.offset[1] - 50), (255, 0, 0), 1, tipLength=0.1)
        cv2.arrowedLine(frame, (center_x + self.offset[0], center_y + self.offset[1]), (center_x + self.offset[0] + 50, center_y + self.offset[1]), (255, 0, 0), 1, tipLength=0.1)
        cv2.circle(frame, (center_x + self.offset[0], center_y + self.offset[1]), 5, (255, 0, 0), -1)
        cv2.circle(frame, (center_x + self.offset[0], center_y + self.offset[1]), self.circle_detect, (255, 0, 0), 2)
        return frame

    def tracking_object(self, frame, detects, id):
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        len_detected = len(detects)
        if len_detected == 0:
            return None
        errors = []
        for detect in range(len_detected):
            x, y, w, h = detects[detect]
            detected_center_x = x + w // 2
            detected_center_y = y + h // 2
            if (detected_center_x in range(center_x + self.offset[0] - self.circle_detect, center_x + self.offset[0] + self.circle_detect)) and \
               (detected_center_y in range(center_y + self.offset[1] - self.circle_detect, center_y + self.offset[1] + self.circle_detect)):
                error_x = float(detected_center_x - (center_x + self.offset[0]))
                error_y = float(detected_center_y - (center_y + self.offset[1]))
                errors.append([float(error_x), float(error_y), id])
        if errors:
            closer_error = min(errors, key=lambda x: abs(x[0]) + abs(x[1]))
            errors = [closer_error]
        return errors

######################
def control_drone_to_center(errors, color_idx):
    i = color_idx * 3
    error_x, error_y, color_id = errors[i:i+3]
    if error_x != 0.0 or error_y != 0.0:
        vx = PID_Y.update(-error_y)
        vy = PID_X.update(error_x)
        print("[PID CONTROL] Sending velocity vx: {:.3f}, vy: {:.3f} for color id".format(vx, vy))
        send_local_ned_velocity(vx, vy, 0)
        return True
    send_local_ned_velocity(0, 0, 0)
    return False

## landing H marker

def detect_H_center(img):
    global MIN_CONTOUR_AREA, ASPECT_RATIO_MIN, ASPECT_RATIO_MAX
    global SOLIDITY_THRESH, HOLE_RATIO_MIN, HOLE_RATIO_MAX

    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours with hierarchy
    _,contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or hierarchy is None:
        return None, None, img

    hierarchy = hierarchy[0]
    best_contour = None
    best_score = 0

    for i, contour in enumerate(contours):
        # Skip inner contours
        if hierarchy[i][3] == -1:
            continue

        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        # Check aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
            continue

        # Check solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / float(hull_area)
        if solidity < SOLIDITY_THRESH:
            continue

        # Check for internal hole (characteristic of H)
        hole_area = 0
        child_idx = hierarchy[i][2]
        if child_idx != -1:
            while child_idx != -1:
                hole_area += cv2.contourArea(contours[child_idx])
                child_idx = hierarchy[child_idx][0]

        hole_ratio = hole_area / float(area)
        if hole_ratio < HOLE_RATIO_MIN or hole_ratio > HOLE_RATIO_MAX:
            continue

        # Calculate score based on geometric properties
        score = solidity + (1 - abs(aspect_ratio - 1))
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None, None, img

    # Calculate centroid
    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        return None, None, img

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Draw contour and center on image
    cv2.drawContours(img, [best_contour], -1, (0, 255, 0), 2)
    cv2.circle(img, (cX, cY), 7, (0, 0, 255), -1)
    cv2.putText(img, "H", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return cX, cY, img




def mission_state_machine(errors, frame=None):
    global mission_step, hold_start_time, in_target_zone, is_detected
    offset = [0,130]
    circle_detect = 250
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    # mission_step:
    # 0 - takeoff and move left, 1 - detect yellow 1st, 2 - left, 3 - detect yellow 2nd,
    # 4 - left, 5 - detect blue, 6 - right+forward, 7 - detect red 1st,
    # 8 - backward, 9 - detect red 2nd, 10 - forward+right (go home), 11 - done

    if mission_step == 0:
        print("Takeoff and move left to start mission.")
        move_with_timer('left', 16 ,speed=1)
        cv2.arrowedLine(frame, (center_x + offset[0], center_y + offset[1]), (center_x + offset[0], center_y + offset[1] - 50), (0, 255, 0), 2, tipLength=0.1)

        mission_step = 1

    elif mission_step == 1:
        # Detect yellow 1st
        if not hasattr(mission_state_machine, "stabilize_drone"):
            mission_state_machine.stabilize_drone = False
            mission_state_machine.stabilize_start_time = 0
        if not mission_state_machine.stabilize_drone:
            print("Stabilizing drone before detecting yellow 1st.")
            send_local_ned_velocity(0, 0, 0)
            if mission_state_machine.stabilize_start_time == 0:
                mission_state_machine.stabilize_start_time = time.time()
            if time.time() - mission_state_machine.stabilize_start_time > 4:
                mission_state_machine.stabilize_drone = True
                print("Drone stabilized, ready to detect yellow 1st.")
            return


        error_x, error_y, _ = errors[6:9]
        is_detected = (error_x != 0.0 or error_y != 0.0)
        if is_detected and is_in_target_zone(error_x, error_y):
            if not in_target_zone:
                in_target_zone = True
                hold_start_time = time.time()
            else:
                elapsed = time.time() - hold_start_time
                if frame is not None:
                    cv2.putText(frame, "Hold Yellow 1: %d / 2 s" % int(elapsed), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                if elapsed > 2:
                    print("Yellow 1 detected. Move left.")
                    move_with_timer('left', 13, speed=0.5)
                    cv2.arrowedLine(frame, (center_x + offset[0], center_y + offset[1]), (center_x + offset[0], center_y + offset[1] - 50), (0, 255, 0), 2, tipLength=0.1)

                    in_target_zone = False
                    mission_step = 2
        else:
            in_target_zone = False
            hold_start_time = 0
        control_drone_to_center(errors, 2)

    elif mission_step == 2:
        print("Moved left, switching to detect yellow 2nd.")
        mission_step = 3

    elif mission_step == 3:
        # Detect yellow 2nd
        error_x, error_y, _ = errors[6:9]
        is_detected = (error_x != 0.0 or error_y != 0.0)
        if is_detected and is_in_target_zone(error_x, error_y):
            if not in_target_zone:
                in_target_zone = True
                hold_start_time = time.time()
            else:
                elapsed = time.time() - hold_start_time
                if frame is not None:
                    cv2.putText(frame, "Hold Yellow 2: %d / 2 s" % int(elapsed), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                if elapsed > 2:
                    print("Yellow 2 detected. Move left.")
                    move_with_timer('left', 13, speed=0.5)
                    in_target_zone = False
                    mission_step = 4
        else:
            in_target_zone = False
            hold_start_time = 0
        control_drone_to_center(errors, 2)

    elif mission_step == 4:
        print("Moved left 2nd, switching to detect blue.")
        mission_step = 5

    elif mission_step == 5:
        # Detect blue
        if not hasattr(mission_state_machine, "stabilize_drone"):
            mission_state_machine.stabilize_drone = False
            mission_state_machine.stabilize_start_time = 0
        if not mission_state_machine.stabilize_drone:
            print("Stabilizing drone before detecting blue .")
            send_local_ned_velocity(0, 0, 0)
            if mission_state_machine.stabilize_start_time == 0:
                mission_state_machine.stabilize_start_time = time.time()
            if time.time() - mission_state_machine.stabilize_start_time > 2:
                mission_state_machine.stabilize_drone = True
                print("Drone stabilized, ready to detect blue.")
            return
        error_x, error_y, _ = errors[3:6]
        is_detected = (error_x != 0.0 or error_y != 0.0)
        if is_detected and is_in_target_zone(error_x, error_y):
            if not in_target_zone:
                in_target_zone = True
                hold_start_time = time.time()
            else:
                elapsed = time.time() - hold_start_time
                if frame is not None:
                    cv2.putText(frame, "Hold Blue: %d / 10 s" % int(elapsed), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                if elapsed > 10:
                    print("Blue detected. Move right + forward.")
                    move_with_timer('forward', 6, speed=0.5)
                    move_with_timer('right', 13, speed=0.5)
                    in_target_zone = False
                    mission_step = 6
        else:
            in_target_zone = False
            hold_start_time = 0
        control_drone_to_center(errors, 1)

    elif mission_step == 6:
        print("Moved right + forward, switching to detect red 1st.")
        mission_step = 7

    elif mission_step == 7:
        # Detect red 1st
        error_x, error_y, _ = errors[0:3]
        is_detected = (error_x != 0.0 or error_y != 0.0)
        if is_detected and is_in_target_zone(error_x, error_y):
            if not in_target_zone:
                in_target_zone = True
                hold_start_time = time.time()
            else:
                elapsed = time.time() - hold_start_time
                if frame is not None:
                    cv2.putText(frame, "Hold Red 1: %d / 5 s" % int(elapsed), (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if elapsed > 5:
                    print("Red 1 detected. Move backward.")
                    move_with_timer('backward', 12, speed=0.6)
                    in_target_zone = False
                    mission_step = 8
        else:
            in_target_zone = False
            hold_start_time = 0
        control_drone_to_center(errors, 0)

    elif mission_step == 8:
        print("Moved backward, switching to detect red 2nd.")
        mission_step = 9

    elif mission_step == 9:
        # Detect red 2nd
        error_x, error_y, _ = errors[0:3]
        is_detected = (error_x != 0.0 or error_y != 0.0)
        if is_detected and is_in_target_zone(error_x, error_y):
            if not in_target_zone:
                in_target_zone = True
                hold_start_time = time.time()
            else:
                elapsed = time.time() - hold_start_time
                if frame is not None:
                    cv2.putText(frame, "Hold Red 2: %d / 5 s" % int(elapsed), (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if elapsed > 5:
                    print("Red 2 detected. Move forward + right (go home).")
                    move_with_timer('forward', 0.5, speed=0.5)
                    move_with_timer('right', 38, speed=0.6)
                    in_target_zone = False
                    mission_step = 10
        else:
            in_target_zone = False
            hold_start_time = 0
        control_drone_to_center(errors, 0)

    elif mission_step == 10:
        print("Mission completed! Drone returned home.")
        if frame is not None:
            np_data = frame.copy()  # use the new frame
        else:
            print("No frame avaiable for landing marker detection")
            return

        cX, cY, out_img = detect_H_center(np_data)

        current_alt = vehicle.location.global_relative_frame.alt
        if cX is not None and cY is not None:
            x_ang = (cX - horizontal_res * 0.5) * horizontal_fov / horizontal_res
            y_ang = (cY - vertical_res * 0.5) * vertical_fov / vertical_res

            if vehicle.mode.name != 'LAND':
                vehicle.mode = VehicleMode('LAND')
                while vehicle.mode.name != 'LAND':
                    time.sleep(0.5)
                print('Vehicle in LAND mode')

            send_land_message(x_ang, y_ang)
            print("Landing target detected at ({}, {}) | x_ang: {:.4f}, y_ang: {:.4f}".format(cX, cY, x_ang, y_ang))

            # Only use out_img if it exists
            if out_img is not None:
                cv2.putText(out_img, "Landing Marker: x={}, y={}".format(cX, cY), (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow("Landing View", out_img)
                cv2.waitKey(1)
        else:
            print("No landing marker (H) detected.")




def msg_receiver(msg):
    global time_last, found_count, notfound_count
    if time.time() - time_last < time_to_wait:
        return
    current_alt = vehicle.location.global_relative_frame.alt

    np_data = rnp.numpify(msg)
    clean_frame = np_data.copy()

    segmenter = ColorSegmenter()
    center_tracker = CenterTracker()

    detected_image_red, detections = segmenter.detect_objects(np_data, "red")
    detected_image_red = center_tracker.draw_center_track(detected_image_red)
    errors_red = center_tracker.tracking_object(detected_image_red, detections, 0.0)

    detected_image_blue, detections = segmenter.detect_objects(np_data, "blue")
    detected_image_blue = center_tracker.draw_center_track(detected_image_blue)
    errors_blue = center_tracker.tracking_object(detected_image_blue, detections, 1.0)

    detected_image_yellow, detections = segmenter.detect_objects(np_data, "yellow")
    detected_image_yellow = center_tracker.draw_center_track(detected_image_yellow)
    errors_yellow = center_tracker.tracking_object(detected_image_yellow, detections, 2.0)

    errors = []
    # red
    if errors_red is not None and len(errors_red) > 0:
        errors.extend(errors_red[0])
    else:
        errors.extend([0.0, 0.0, 0.0])
    # blue
    if errors_blue is not None and len(errors_blue) > 0:
        errors.extend(errors_blue[0])
    else:
        errors.extend([0.0, 0.0, 1.0])
    # yellow
    if errors_yellow is not None and len(errors_yellow) > 0:
        errors.extend(errors_yellow[0])
    else:
        PID_X.reset()
        PID_Y.reset()
        errors.extend([0.0, 0.0, 2.0])

    # State machine mission
    if mission_step == 10:
        mission_state_machine(errors, clean_frame)
        cv2.imshow("Landing - H Marker", clean_frame)
    else:
        mission_state_machine(errors, detected_image_yellow)
        cv2.imshow("Detect Objects Red - Blue - Yellow", detected_image_yellow)
    cv2.waitKey(1)

    new_msg = rnp.msgify(Image, np_data, encoding='rgb8')
    newimg_pub.publish(new_msg)
    time_last = time.time()


def subscriber():
    rospy.init_node('drone_node', anonymous=False)
    rospy.Subscriber('/camera/color/image_raw', Image, msg_receiver)
    rospy.spin()

if __name__ == '__main__':
    try:
        arm_and_takeoff(takeoff_height)
        time.sleep(1)
        subscriber()
    except rospy.ROSInterruptException:
        vehicle.close()
        print("Mission completed")