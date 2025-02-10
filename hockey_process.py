#!/usr/bin/python3

import sys
import os
import threading
import queue
import time

from math import *
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

import cv2
import numpy as np
from ultralytics import YOLO

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QPushButton, QRadioButton, QFileDialog, QLineEdit, QCheckBox, QMessageBox, QProgressBar
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt6.uic import loadUi

dir_path = os.path.dirname(os.path.realpath(__file__))

# constants:

FPS = 60
CONFIDENCE_THRESHOLD = 0.5 # to use in make_field_mask and process_hit_ball
HIT_BALL_PADDING = 20

HOCKEY_FIELD_IMAGE_PATH = os.path.join(dir_path, "images", "hockey_field.jpg")
UI_FILE_PATH = os.path.join(dir_path, "main.ui")
MAIN_MODEL_PATH = os.path.join(dir_path, "models/main.pt")
KEYPOINTS_MODEL_PATH = os.path.join(dir_path, "models/keypoints.pt")
BALL_MODEL_PATH = os.path.join(dir_path, "models/ball.pt")
CLASSIFY_MODEL_PATH = os.path.join(dir_path, "models/classify.pt")

# end



# classes in YOLO model
KP_NOTCH_CLASS = 0
KP_PERP_T_FAR_CLASS = 1
KP_PERP_T_NEAR_CLASS = 2
KP_ARC_T_LEFT_FAR_CLASS = 3
KP_ARC_T_LEFT_NEAR_CLASS = 4
KP_ARC_T_RIGHT_FAR_CLASS = 5
KP_ARC_T_RIGHT_NEAR_CLASS = 6
KP_CORNER_LEFT_FAR_CLASS = 7
KP_CORNER_LEFT_NEAR_CLASS = 8
KP_CORNER_RIGHT_FAR_CLASS = 9
KP_CORNER_RIGHT_NEAR_CLASS = 10
KP_GATE_CENTER_LEFT_CLASS = 11
KP_GATE_CENTER_RIGHT_CLASS = 12
KP_ARC_CENTER_LEFT_CLASS = 13
KP_ARC_CENTER_RIGHT_CLASS = 14
KP_ARC_POINT_MARK_CLASS = 15

# assume we look from center of wide side:
KP_PERP_T_FAR_LEFT = 0
KP_PERP_T_FAR_MID = 1
KP_PERP_T_FAR_RIGHT = 2
KP_PERP_T_NEAR_LEFT = 3
KP_PERP_T_NEAR_MID = 4
KP_PERP_T_NEAR_RIGHT = 5
KP_ARC_T_LEFT_FAR = 6
KP_ARC_T_LEFT_NEAR = 7
KP_ARC_T_RIGHT_FAR = 8
KP_ARC_T_RIGHT_NEAR = 9
KP_CORNER_LEFT_FAR = 10
KP_CORNER_LEFT_NEAR = 11
KP_CORNER_RIGHT_FAR = 12
KP_CORNER_RIGHT_NEAR = 13
KP_GATE_CENTER_LEFT = 14
KP_GATE_CENTER_RIGHT = 15
KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST = 16
KP_NOTCH_WIDE_FAR_LEFT_CENTERMOST = 17
KP_NOTCH_WIDE_FAR_RIGHT_CORNERMOST = 18
KP_NOTCH_WIDE_FAR_RIGHT_CENTERMOST = 19
KP_NOTCH_WIDE_NEAR_LEFT_CORNERMOST = 20
KP_NOTCH_WIDE_NEAR_LEFT_CENTERMOST = 21
KP_NOTCH_WIDE_NEAR_RIGHT_CORNERMOST = 22
KP_NOTCH_WIDE_NEAR_RIGHT_CENTERMOST = 23
KP_NOTCH_NARROW_LEFT_FAR_ARCMOST = 24
KP_NOTCH_NARROW_LEFT_FAR_GATEMOST = 25
KP_NOTCH_NARROW_LEFT_NEAR_ARCMOST = 26
KP_NOTCH_NARROW_LEFT_NEAR_GATEMOST = 27
KP_NOTCH_NARROW_RIGHT_FAR_ARCMOST = 28
KP_NOTCH_NARROW_RIGHT_FAR_GATEMOST = 29
KP_NOTCH_NARROW_RIGHT_NEAR_ARCMOST = 30
KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST = 31
KP_ARC_CENTER_LEFT = 32
KP_ARC_CENTER_RIGHT = 33
KP_ARC_POINT_MARK_LEFT = 34
KP_ARC_POINT_MARK_RIGHT = 35

def kp_name(kp):
	return {
		KP_PERP_T_FAR_LEFT: 'pt_far_left',
		KP_PERP_T_FAR_MID: 'pt_far_mid',
		KP_PERP_T_FAR_RIGHT: 'pt_far_right',
		KP_PERP_T_NEAR_LEFT: 'pt_near_left',
		KP_PERP_T_NEAR_MID: 'pt_near_mid',
		KP_PERP_T_NEAR_RIGHT: 'pt_near_right',
		KP_ARC_T_LEFT_FAR: 'at_left_far',
		KP_ARC_T_LEFT_NEAR: 'at_left_near',
		KP_ARC_T_RIGHT_FAR: 'at_right_far',
		KP_ARC_T_RIGHT_NEAR: 'at_right_near',
		KP_CORNER_LEFT_FAR: 'corn_left_far',
		KP_CORNER_LEFT_NEAR: 'corn_left_near',
		KP_CORNER_RIGHT_FAR: 'corn_right_far',
		KP_CORNER_RIGHT_NEAR: 'corn_right_near',
		KP_GATE_CENTER_LEFT: 'gate_center_left',
		KP_GATE_CENTER_RIGHT: 'gate_center_right',
		KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST: 'n_far_left_corner',
		KP_NOTCH_WIDE_FAR_LEFT_CENTERMOST: 'n_far_left_center',
		KP_NOTCH_WIDE_FAR_RIGHT_CORNERMOST: 'n_far_right_corner',
		KP_NOTCH_WIDE_FAR_RIGHT_CENTERMOST: 'n_far_right_center',
		KP_NOTCH_WIDE_NEAR_LEFT_CORNERMOST: 'n_near_left_corner',
		KP_NOTCH_WIDE_NEAR_LEFT_CENTERMOST: 'n_near_left_center',
		KP_NOTCH_WIDE_NEAR_RIGHT_CORNERMOST: 'n_near_right_corner',
		KP_NOTCH_WIDE_NEAR_RIGHT_CENTERMOST: 'n_near_right_center',
		KP_NOTCH_NARROW_LEFT_FAR_ARCMOST: 'n_left_far_arc',
		KP_NOTCH_NARROW_LEFT_FAR_GATEMOST: 'n_left_far_gate',
		KP_NOTCH_NARROW_LEFT_NEAR_ARCMOST: 'n_left_near_arc',
		KP_NOTCH_NARROW_LEFT_NEAR_GATEMOST: 'n_left_near_gate',
		KP_NOTCH_NARROW_RIGHT_FAR_ARCMOST: 'n_right_far_arc',
		KP_NOTCH_NARROW_RIGHT_FAR_GATEMOST: 'n_right_far_gate',
		KP_NOTCH_NARROW_RIGHT_NEAR_ARCMOST: 'n_right_near_arc',
		KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST: 'n_right_near_gate',
		KP_ARC_CENTER_LEFT: 'arc_center_left',
		KP_ARC_CENTER_RIGHT: 'arc_center_right',
		KP_ARC_POINT_MARK_LEFT: 'point_left',
		KP_ARC_POINT_MARK_RIGHT: 'point_right',
	}[kp]

def kp_color(kp):
	red = 160
	green = 15
	blue = 160
	
	if kp >= KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST and kp <= KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST:
		rem = KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST - KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST
		value = int(((kp - KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST)%rem)/float(rem) * 127.0)
		
		green = 128 + value
		blue = 64 + value
		
		red = 15
	else:
		if kp < 20:
			red += int((kp%20)/20.0 * 95.0)
		else:
			blue += int(((kp - 20)%20)/20.0 * 95.0)
	
	return (red, green, blue)



LINE_DISTANCE_MAX = 0.015

def point_is_near_line(p0, p1, point, threshold):
	line_vector = p1 - p0
	line_normal = np.array([-line_vector[1], line_vector[0]])
	
	distance_to_line = np.abs(np.dot(point - p0, line_normal)) / np.linalg.norm(line_vector)
	
	line_length_squared = np.dot(line_vector, line_vector)
	projection_factor = np.dot(point - p0, line_vector) / line_length_squared
	
	return (distance_to_line <= LINE_DISTANCE_MAX and projection_factor >= 0.0 and projection_factor <= 1.0, projection_factor)

def map_notchlike(real_keypoints, in_kp, notchlike_class, notchlike_ids, start_id, end_id):
	if start_id in real_keypoints and end_id in real_keypoints:
		p0 = real_keypoints[start_id]
		p1 = real_keypoints[end_id]
		
		add_keypoints = []
		
		for kp in in_kp:
			cls = kp[0]
			coords = kp[1]
			
			if cls == notchlike_class:
				if point_is_near_line(p0, p1, coords, LINE_DISTANCE_MAX)[0]:
					add_keypoints.append(coords)
		
		if len(add_keypoints) != 0:
			indices = np.argsort(np.linalg.norm(add_keypoints - p0, axis=1))
			
			if len(indices) == len(notchlike_ids):
				for i in range(len(notchlike_ids)):
					real_keypoints[notchlike_ids[i]] = add_keypoints[indices[i]]

MAX_TRACK_DISTANCE = 0.025

def track_keypoint(src_coords, in_kp, test_cls):
	for kp in in_kp:
		cls = kp[0]
		coords = kp[1]
		
		if cls == test_cls:
			distance = np.linalg.norm(src_coords - coords)
			
			if distance <= MAX_TRACK_DISTANCE:
				return coords
	
	return None

MAX_CONFLICT_DISTANCE = 0.025

def solve_side_conflicts(real_keypoints, is_left, is_right, left_ids, right_ids):
	for (left_id, right_id) in zip(left_ids, right_ids):
		if left_id in real_keypoints and right_id in real_keypoints:
			left_coords = real_keypoints[left_id]
			right_coords = real_keypoints[right_id]
			
			distance = np.linalg.norm(left_coords - right_coords)
			
			if distance <= MAX_CONFLICT_DISTANCE:
				if not is_left:
					del real_keypoints[left_id]
				if not is_right:
					del real_keypoints[right_id]

def set_perps(real_keypoints, perps, is_left, is_right, left_id, mid_id, right_id):
	if len(perps) == 1: # only extreme
		if is_left:
			real_keypoints[left_id] = perps[0]
		elif is_right:
			real_keypoints[right_id] = perps[0]
	elif len(perps) == 2: # extreme and middle
		leftmost_index = np.argmin(perps[:, 0])
		rightmost_index = np.argmax(perps[:, 0])
		
		if is_left:
			real_keypoints[left_id] = perps[leftmost_index]
			real_keypoints[mid_id] = perps[rightmost_index]
		elif is_right:
			real_keypoints[right_id] = perps[rightmost_index]
		real_keypoints[mid_id] = perps[leftmost_index]
	elif len(perps) == 3:
		indices = np.argsort(perps[:, 0])
		
		real_keypoints[left_id] = perps[indices[0]]
		real_keypoints[mid_id] = perps[indices[1]]
		real_keypoints[right_id] = perps[indices[2]]

# in_kp: [(class, np.array([x, y])), ...]; x and y normalized to [0.0, 1.0]
def guess_real_keypoints(in_kp, prev_keypoints):
	real_keypoints = {} # KP_ID: (x, y)
	
	
	# at first take all solid keypoints
	for kp in in_kp:
		cls = kp[0]
		coords = kp[1]
		
		if cls == KP_ARC_T_LEFT_FAR_CLASS:
			real_keypoints[KP_ARC_T_LEFT_FAR] = coords
		elif cls == KP_ARC_T_LEFT_NEAR_CLASS:
			real_keypoints[KP_ARC_T_LEFT_NEAR] = coords
		elif cls == KP_ARC_T_RIGHT_FAR_CLASS:
			real_keypoints[KP_ARC_T_RIGHT_FAR] = coords
		elif cls == KP_ARC_T_RIGHT_NEAR_CLASS:
			real_keypoints[KP_ARC_T_RIGHT_NEAR] = coords
		elif cls == KP_CORNER_LEFT_FAR_CLASS:
			real_keypoints[KP_CORNER_LEFT_FAR] = coords
		elif cls == KP_CORNER_LEFT_NEAR_CLASS:
			real_keypoints[KP_CORNER_LEFT_NEAR] = coords
		elif cls == KP_CORNER_RIGHT_FAR_CLASS:
			real_keypoints[KP_CORNER_RIGHT_FAR] = coords
		elif cls == KP_CORNER_RIGHT_NEAR_CLASS:
			real_keypoints[KP_CORNER_RIGHT_NEAR] = coords
	
	
	likely_left = KP_CORNER_LEFT_FAR in real_keypoints or KP_CORNER_LEFT_NEAR in real_keypoints or KP_ARC_T_LEFT_FAR in real_keypoints or KP_ARC_T_LEFT_NEAR in real_keypoints
	likely_right = KP_CORNER_RIGHT_FAR in real_keypoints or KP_CORNER_RIGHT_NEAR in real_keypoints or KP_ARC_T_RIGHT_FAR in real_keypoints or KP_ARC_T_RIGHT_NEAR in real_keypoints
	
	# set ARC_CENTER, GATE_CENTER and ARC_POINT_MARK
	if likely_left != likely_right:
		for kp in in_kp:
			cls = kp[0]
			coords = kp[1]
			
			if cls == KP_ARC_CENTER_LEFT_CLASS or cls == KP_ARC_CENTER_RIGHT_CLASS:
				if likely_left:
					real_keypoints[KP_ARC_CENTER_LEFT] = coords
				elif likely_right:
					real_keypoints[KP_ARC_CENTER_RIGHT] = coords
			elif cls == KP_GATE_CENTER_LEFT_CLASS or cls == KP_GATE_CENTER_RIGHT_CLASS:
				if likely_left:
					real_keypoints[KP_GATE_CENTER_LEFT] = coords
				elif likely_right:
					real_keypoints[KP_GATE_CENTER_RIGHT] = coords
			elif cls == KP_ARC_POINT_MARK_CLASS:
				if likely_left:
					real_keypoints[KP_ARC_POINT_MARK_LEFT] = coords
				elif likely_right:
					real_keypoints[KP_ARC_POINT_MARK_RIGHT] = coords
	
	
	
	# assuming, that when we see left side and don't see right side, then any PERP_T keypoint will be at that side
	is_left = likely_left or (KP_ARC_CENTER_LEFT in real_keypoints or KP_GATE_CENTER_LEFT in real_keypoints)
	is_right = likely_right or (KP_ARC_CENTER_RIGHT in real_keypoints or KP_GATE_CENTER_RIGHT in real_keypoints)
	
	far_perps = []
	near_perps = []
	for kp in in_kp:
		cls = kp[0]
		coords = kp[1]
		
		if cls == KP_PERP_T_FAR_CLASS:
			far_perps.append(coords)
		elif cls == KP_PERP_T_NEAR_CLASS:
			near_perps.append(coords)
	
	if is_left != is_right: # != is XOR
		set_perps(real_keypoints, np.array(far_perps), is_left, is_right, KP_PERP_T_FAR_LEFT, KP_PERP_T_FAR_MID, KP_PERP_T_FAR_RIGHT)
		set_perps(real_keypoints, np.array(near_perps), is_left, is_right, KP_PERP_T_NEAR_LEFT, KP_PERP_T_NEAR_MID, KP_PERP_T_NEAR_RIGHT)
	else:
		if len(far_perps) == 3:
			set_perps(real_keypoints, np.array(far_perps), is_left, is_right, KP_PERP_T_FAR_LEFT, KP_PERP_T_FAR_MID, KP_PERP_T_FAR_RIGHT)
		if len(near_perps) == 3:
			set_perps(real_keypoints, np.array(near_perps), is_left, is_right, KP_PERP_T_NEAR_LEFT, KP_PERP_T_NEAR_MID, KP_PERP_T_NEAR_RIGHT)
	
	
	# find corresponding notches
	# wide left
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST, KP_NOTCH_WIDE_FAR_LEFT_CENTERMOST], KP_CORNER_LEFT_FAR, KP_PERP_T_FAR_LEFT)
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_WIDE_NEAR_LEFT_CORNERMOST, KP_NOTCH_WIDE_NEAR_LEFT_CENTERMOST], KP_CORNER_LEFT_NEAR, KP_PERP_T_NEAR_LEFT)
	
	# narrow left
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_NARROW_LEFT_FAR_ARCMOST, KP_NOTCH_NARROW_LEFT_FAR_GATEMOST], KP_ARC_T_LEFT_FAR, KP_GATE_CENTER_LEFT)
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_NARROW_LEFT_NEAR_ARCMOST, KP_NOTCH_NARROW_LEFT_NEAR_GATEMOST], KP_ARC_T_LEFT_NEAR, KP_GATE_CENTER_LEFT)
	
	# wide right
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_WIDE_FAR_RIGHT_CORNERMOST, KP_NOTCH_WIDE_FAR_RIGHT_CENTERMOST], KP_CORNER_RIGHT_FAR, KP_PERP_T_FAR_RIGHT)
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_WIDE_NEAR_RIGHT_CORNERMOST, KP_NOTCH_WIDE_NEAR_RIGHT_CENTERMOST], KP_CORNER_RIGHT_NEAR, KP_PERP_T_NEAR_RIGHT)
	
	# narrow right
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_NARROW_RIGHT_FAR_ARCMOST, KP_NOTCH_NARROW_RIGHT_FAR_GATEMOST], KP_ARC_T_RIGHT_FAR, KP_GATE_CENTER_RIGHT)
	map_notchlike(real_keypoints, in_kp, KP_NOTCH_CLASS, [KP_NOTCH_NARROW_RIGHT_NEAR_ARCMOST, KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST], KP_ARC_T_RIGHT_NEAR, KP_GATE_CENTER_RIGHT)
	
	
	# then to track old ones which aren't recognized to this point
	if prev_keypoints is not None:
		for idx, coords in prev_keypoints.items():
			if idx not in real_keypoints:
				test_cls = None
				if idx in [KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST, KP_NOTCH_WIDE_FAR_LEFT_CENTERMOST, KP_NOTCH_WIDE_FAR_RIGHT_CORNERMOST, \
					       KP_NOTCH_WIDE_FAR_RIGHT_CENTERMOST, KP_NOTCH_WIDE_NEAR_LEFT_CORNERMOST, KP_NOTCH_WIDE_NEAR_LEFT_CENTERMOST, \
					       KP_NOTCH_WIDE_NEAR_RIGHT_CORNERMOST, KP_NOTCH_WIDE_NEAR_RIGHT_CENTERMOST, KP_NOTCH_NARROW_LEFT_FAR_ARCMOST, \
					       KP_NOTCH_NARROW_LEFT_FAR_GATEMOST, KP_NOTCH_NARROW_LEFT_NEAR_ARCMOST, KP_NOTCH_NARROW_LEFT_NEAR_GATEMOST, \
					       KP_NOTCH_NARROW_RIGHT_FAR_ARCMOST, KP_NOTCH_NARROW_RIGHT_FAR_GATEMOST, KP_NOTCH_NARROW_RIGHT_NEAR_ARCMOST, \
					       KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST]:
					
					test_cls = KP_NOTCH_CLASS
				
				elif idx in [KP_PERP_T_FAR_LEFT, KP_PERP_T_FAR_MID, KP_PERP_T_FAR_RIGHT]:
					test_cls = KP_PERP_T_FAR_CLASS
				
				elif idx in [KP_PERP_T_NEAR_LEFT, KP_PERP_T_NEAR_MID, KP_PERP_T_NEAR_RIGHT]:
					test_cls = KP_PERP_T_NEAR_CLASS
				
				elif idx in [KP_ARC_POINT_MARK_LEFT, KP_ARC_POINT_MARK_RIGHT]:
					test_cls = KP_ARC_POINT_MARK_CLASS
				
				if test_cls is not None:
					coords = track_keypoint(coords, in_kp, test_cls)
					
					if coords is not None:
						real_keypoints[idx] = coords
	
	
	# if left and right keypoints are on the same spot, then delete one or another or both
	solve_side_conflicts(real_keypoints, is_left, is_right, \
	    [KP_ARC_CENTER_LEFT, KP_GATE_CENTER_LEFT, KP_ARC_T_LEFT_NEAR, KP_ARC_T_LEFT_FAR, KP_NOTCH_NARROW_LEFT_FAR_ARCMOST, KP_NOTCH_NARROW_LEFT_FAR_GATEMOST, KP_NOTCH_NARROW_LEFT_NEAR_ARCMOST, KP_NOTCH_NARROW_LEFT_NEAR_GATEMOST], \
	    [KP_ARC_CENTER_RIGHT, KP_GATE_CENTER_RIGHT, KP_ARC_T_RIGHT_NEAR, KP_ARC_T_RIGHT_FAR, KP_NOTCH_NARROW_RIGHT_FAR_ARCMOST, KP_NOTCH_NARROW_RIGHT_FAR_GATEMOST, KP_NOTCH_NARROW_RIGHT_NEAR_ARCMOST, KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST]
	)
	
	
	return real_keypoints





def angle_between_vectors(v1, v2):
	cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	cos_theta = np.clip(cos_theta, -1, 1)
	angle = np.arccos(cos_theta)
	return angle

QUADRILATERAL_ANGLE_MAX = radians(160)

def is_valid_quadrilateral(points):
	center = np.mean(points, axis=0)
	sorted_points = sorted(points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
	
	angles = []
	for i in range(4):
		p1 = sorted_points[(i - 1) % 4]  # prev point
		p2 = sorted_points[i]            # current point
		p3 = sorted_points[(i + 1) % 4]  # next point
		v1 = p1 - p2
		v2 = p3 - p2
		angle = angle_between_vectors(v1, v2)
		
		if angle <= 0.0 or angle >= QUADRILATERAL_ANGLE_MAX:
			return False
	
	return True

def keypoints_has_enough_data(keypoints):
	if len(keypoints) < 4:
		return False
	
	points = [keypoints[key] for key in keypoints]
	
	for quad in combinations(points, 4):
		quad = np.array(quad)
		if is_valid_quadrilateral(quad):
			return True
	
	return False



# coords (x, y) are in some units, where (0, 0) is the center of the hockey field image, X+ is right and Y+ is up

# here units are meters

FIELD_WIDE_Y = 27.51
FIELD_NARROW_X = 45.72
FIELD_ARC_CENTER_X = 31.09
FIELD_PERP_T_SIDE_X = 22.86
FIELD_ARC_T_Y = 16.82
FIELD_NOTCH_CORNERMOST_X = 40.72
FIELD_NOTCH_CENTERMOST_X = 31.09
FIELD_NOTCH_ARCMOST_Y = 11.83
FIELD_NOTCH_GATEMOST_Y = 6.83
FIELD_ARC_POINT_MARK_X = 39.32

FIELD_KP_COORDS = {
	KP_PERP_T_FAR_LEFT :                  (-FIELD_PERP_T_SIDE_X, FIELD_WIDE_Y),
	KP_PERP_T_FAR_MID :                   (0.0, FIELD_WIDE_Y),
	KP_PERP_T_FAR_RIGHT :                 (FIELD_PERP_T_SIDE_X, FIELD_WIDE_Y),
	KP_PERP_T_NEAR_LEFT :                 (-FIELD_PERP_T_SIDE_X, -FIELD_WIDE_Y),
	KP_PERP_T_NEAR_MID :                  (0.0, -FIELD_WIDE_Y),
	KP_PERP_T_NEAR_RIGHT :                (FIELD_PERP_T_SIDE_X, -FIELD_WIDE_Y),
	
	KP_ARC_T_LEFT_FAR :                   (-FIELD_NARROW_X, FIELD_ARC_T_Y),
	KP_ARC_T_LEFT_NEAR :                  (-FIELD_NARROW_X, -FIELD_ARC_T_Y),
	KP_ARC_T_RIGHT_FAR :                  (FIELD_NARROW_X, FIELD_ARC_T_Y),
	KP_ARC_T_RIGHT_NEAR :                 (FIELD_NARROW_X, -FIELD_ARC_T_Y),
	
	KP_CORNER_LEFT_FAR :                  (-FIELD_NARROW_X, FIELD_WIDE_Y),
	KP_CORNER_LEFT_NEAR :                 (-FIELD_NARROW_X, -FIELD_WIDE_Y),
	KP_CORNER_RIGHT_FAR :                 (FIELD_NARROW_X, FIELD_WIDE_Y),
	KP_CORNER_RIGHT_NEAR :                (FIELD_NARROW_X, -FIELD_WIDE_Y),
	
	KP_GATE_CENTER_LEFT :                 (-FIELD_NARROW_X, 0.0),
	KP_GATE_CENTER_RIGHT :                (FIELD_NARROW_X, 0.0),
	
	KP_NOTCH_WIDE_FAR_LEFT_CORNERMOST :   (-FIELD_NOTCH_CORNERMOST_X, FIELD_WIDE_Y),
	KP_NOTCH_WIDE_FAR_LEFT_CENTERMOST :   (-FIELD_NOTCH_CENTERMOST_X, FIELD_WIDE_Y),
	KP_NOTCH_WIDE_FAR_RIGHT_CORNERMOST :  (FIELD_NOTCH_CORNERMOST_X, FIELD_WIDE_Y),
	KP_NOTCH_WIDE_FAR_RIGHT_CENTERMOST :  (FIELD_NOTCH_CENTERMOST_X, FIELD_WIDE_Y),
	KP_NOTCH_WIDE_NEAR_LEFT_CORNERMOST :  (-FIELD_NOTCH_CORNERMOST_X, -FIELD_WIDE_Y),
	KP_NOTCH_WIDE_NEAR_LEFT_CENTERMOST :  (-FIELD_NOTCH_CENTERMOST_X, -FIELD_WIDE_Y),
	KP_NOTCH_WIDE_NEAR_RIGHT_CORNERMOST : (FIELD_NOTCH_CORNERMOST_X, -FIELD_WIDE_Y),
	KP_NOTCH_WIDE_NEAR_RIGHT_CENTERMOST : (FIELD_NOTCH_CENTERMOST_X, -FIELD_WIDE_Y),
	
	KP_NOTCH_NARROW_LEFT_FAR_ARCMOST :    (-FIELD_NARROW_X, FIELD_NOTCH_ARCMOST_Y),
	KP_NOTCH_NARROW_LEFT_FAR_GATEMOST :   (-FIELD_NARROW_X, FIELD_NOTCH_GATEMOST_Y),
	KP_NOTCH_NARROW_LEFT_NEAR_ARCMOST :   (-FIELD_NARROW_X, -FIELD_NOTCH_ARCMOST_Y),
	KP_NOTCH_NARROW_LEFT_NEAR_GATEMOST :  (-FIELD_NARROW_X, -FIELD_NOTCH_GATEMOST_Y),
	KP_NOTCH_NARROW_RIGHT_FAR_ARCMOST :   (FIELD_NARROW_X, FIELD_NOTCH_ARCMOST_Y),
	KP_NOTCH_NARROW_RIGHT_FAR_GATEMOST :  (FIELD_NARROW_X, FIELD_NOTCH_GATEMOST_Y),
	KP_NOTCH_NARROW_RIGHT_NEAR_ARCMOST :  (FIELD_NARROW_X, -FIELD_NOTCH_ARCMOST_Y),
	KP_NOTCH_NARROW_RIGHT_NEAR_GATEMOST : (FIELD_NARROW_X, -FIELD_NOTCH_GATEMOST_Y),
	
	KP_ARC_CENTER_LEFT :                  (-FIELD_ARC_CENTER_X, 0),
	KP_ARC_CENTER_RIGHT :                 (FIELD_ARC_CENTER_X, 0),
	
	KP_ARC_POINT_MARK_LEFT :              (-FIELD_ARC_POINT_MARK_X, 0),
	KP_ARC_POINT_MARK_RIGHT :             (FIELD_ARC_POINT_MARK_X, 0),
}

FIELD_IMAGE_SCALE = 12.85 # pixels per unit



VARIANT_VIDEO = "video"
VARIANT_PLAYERS = "players"
VARIANT_HIT = "hit"
VARIANT_FIELD = "field"
VARIANT_KEYPOINTS = "keypoints"


FIELD_CLASS = 0
ATTACKER_CLASS = 1
KEEPER_CLASS = 2
ANOTHER_PERSON_CLASS = 3

def make_field_mask(frame, result):
	field_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
	
	if result.masks is not None:
		for n, mask in enumerate(result.masks):
			xy = np.array(mask.xy, dtype=np.int32)
			
			if result.boxes.cls[n] == FIELD_CLASS and result.boxes.conf[n] >= CONFIDENCE_THRESHOLD and len(xy[0]) > 0:
				cv2.fillPoly(field_mask, xy, color=(255, 255, 255))
	
	
	player_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
	
	if result.masks is not None:
		for n, mask in enumerate(result.masks):
			xy = np.array(mask.xy, dtype=np.int32)
			
			if result.boxes.cls[n] != FIELD_CLASS and result.boxes.conf[n] >= CONFIDENCE_THRESHOLD and len(xy[0]) > 0:
				cv2.fillPoly(player_mask, xy, color=(255, 255, 255))
	
	
	final_mask = cv2.subtract(field_mask, player_mask)
	
	return final_mask

def process_hit_ball(frame, result, classify_model):
	if result.boxes is not None:
		boxes = [] # [(x0, y0, x1, y1), ...]
		
		for n, box in enumerate(result.boxes):
			if result.boxes.cls[n] == ATTACKER_CLASS and result.boxes.conf[n] >= CONFIDENCE_THRESHOLD:
				x0, y0, x1, y1 = box.cpu().xyxy[0]
				x0 = np.clip(int(x0) - HIT_BALL_PADDING, 0, frame.shape[1] - 1)
				x1 = np.clip(int(x1) + HIT_BALL_PADDING, 0, frame.shape[1] - 1)
				y0 = np.clip(int(y0) - HIT_BALL_PADDING, 0, frame.shape[0] - 1)
				y1 = np.clip(int(y1) + HIT_BALL_PADDING, 0, frame.shape[0] - 1)
				
				subimage = frame[y0:y1, x0:x1]
				
				cls_res = classify_model.predict(subimage)[0]
				
				if cls_res.names[np.argmax(list(cls_res.probs.data.cpu()))] == 'has': # 'has' or 'empty'
					boxes.append((x0, y0, x1, y1))
		
		annotated_frame = frame.copy()
		
		for box in boxes:
			cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
		
		return annotated_frame
	else:
		return frame

def guessed_keypoints_from_result(result, prev_keypoints, frame_width, frame_height):
	result_keypoints = []
	
	boxes = result.boxes.xywh.cpu()
	
	for n, (x, y, _, _) in enumerate(boxes):
		cls = int(result.boxes.cls[n])
		
		result_keypoints.append((cls, np.array([x / frame_width, y / frame_height], dtype=np.float32)))
	
	return guess_real_keypoints(result_keypoints, prev_keypoints)


LINE_KP_MAX_DISTANCE = 0.01
LINE_END_KP_MAX_DISTANCE = 0.015

KL_LINE_BORDER_0 = 0
KL_LINE_BORDER_1 = 1
KL_LEFT = 2
KL_MID = 3
KL_RIGHT = 4

FIELD_KL_VALUES = {
	KL_LEFT: (1.0, 0.0, FIELD_PERP_T_SIDE_X),
	KL_MID: (1.0, 0.0, 0.0),
	KL_RIGHT: (1.0, 0.0, -FIELD_PERP_T_SIDE_X),
}

# ax + by + c = 0
def line_values(p0, p1):
	x0, y0 = p0
	x1, y1 = p1
	
	a = y1 - y0
	b = x0 - x1
	c = -(a * x0 + b * y0)
	
	return (a, b, c)

LINE_VALUES_MAX_SIMILARITY = 0.02

def line_values_similar(v0, v1):
	return abs(v0[0] - v1[0]) <= LINE_VALUES_MAX_SIMILARITY and abs(v0[1] - v1[1]) <= LINE_VALUES_MAX_SIMILARITY and abs(v0[2] - v1[2]) <= LINE_VALUES_MAX_SIMILARITY

def guess_lines(keypoints, frame):
	frame_height, frame_width, _ = frame.shape
	
	_, thresh = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY)
	
	lines_p = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=120, minLineLength=30, maxLineGap=35)
	
	all_lines = []
	
	perp_keys = []
	perps = []
	
	for key in keypoints:
		if key in [KP_PERP_T_FAR_LEFT, KP_PERP_T_FAR_MID, KP_PERP_T_FAR_RIGHT, KP_PERP_T_NEAR_LEFT, KP_PERP_T_NEAR_MID, KP_PERP_T_NEAR_RIGHT]:
			perp_keys.append(key)
			perps.append(keypoints[key])
	
	perp_t_cross = {} # {index : [(p0, p1), ...], ...}
	
	if lines_p is not None:
		for i, perp in enumerate(perps):
			for x0, y0, x1, y1 in lines_p[:, 0]:
				x0 /= frame_width
				y0 /= frame_height
				x1 /= frame_width
				y1 /= frame_height
				
				p0 = np.array([x0, y0])
				p1 = np.array([x1, y1])
				
				all_lines.append((p0, p1))
				
				if point_is_near_line(p0, p1, perp, LINE_KP_MAX_DISTANCE)[0] or np.linalg.norm(perp - p0) <= LINE_END_KP_MAX_DISTANCE or np.linalg.norm(perp - p1) <= LINE_END_KP_MAX_DISTANCE:
					if not i in perp_t_cross:
						perp_t_cross[i] = []
					
					perp_t_cross[i].append((p0, p1))
	
	
	guessed_lines = {} # {ID: [a, b, c], ...} for ax + by + c = 0
	
	line_borders = [] # [(a, b, c)]
	
	for index in perp_t_cross:
		for (p0, p1) in perp_t_cross[index]:
			near, factor = point_is_near_line(p0, p1, perps[index], LINE_KP_MAX_DISTANCE)
			if near and factor >= 0.05 and factor <= 0.95: # then it's a border line
				line_borders.append(line_values(p0, p1))
	
	for index in perp_t_cross:
		for (p0, p1) in perp_t_cross[index]:
			values = line_values(p0, p1)
			
			for border in line_borders:
				if line_values_similar(values, border):
					break
			else:
				key = perp_keys[index]
				
				if key in [KP_PERP_T_FAR_LEFT, KP_PERP_T_NEAR_LEFT]:
					guessed_lines[KL_LEFT] = values
				elif key in [KP_PERP_T_FAR_MID, KP_PERP_T_NEAR_MID]:
					guessed_lines[KL_MID] = values
				elif key in [KP_PERP_T_FAR_RIGHT, KP_PERP_T_NEAR_RIGHT]:
					guessed_lines[KL_RIGHT] = values
	
	if len(line_borders) != 0:
		first = line_borders[0]
		
		guessed_lines[KL_LINE_BORDER_0] = first
		
		for border in line_borders[1:]:
			if not line_values_similar(first, border):
				guessed_lines[KL_LINE_BORDER_1] = border
				break
	
	return all_lines, guessed_lines

def kl_color(kl):
	if kl == KL_LINE_BORDER_0 or kl == KL_LINE_BORDER_1:
		return (255, 0, 0)
	elif kl == KL_LEFT:
		return (255, 127, 0)
	elif kl == KL_MID:
		return (255, 255, 0)
	elif kl == KL_RIGHT:
		return (0, 255, 127)


def data_lines(guessed_lines):
	data_lines = guessed_lines.copy()
	
	if KL_LINE_BORDER_0 in data_lines:
		del data_lines[KL_LINE_BORDER_0]
	if KL_LINE_BORDER_1 in data_lines:
		del data_lines[KL_LINE_BORDER_1]
	
	return data_lines


def compute_homography(points_image=None, points_field=None, lines_image=None, lines_field=None):
	A = []
	
	if points_image is not None and points_field is not None:
		n_points = points_image.shape[0]
		
		for i in range(n_points):
			x, y = points_image[i]
			x_p, y_p = points_field[i]
			A.append([-x, -y, -1,  0,  0,  0, x * x_p, y * x_p, x_p])
			A.append([ 0,  0,  0, -x, -y, -1, x * y_p, y * y_p, y_p])
	
	'''if lines_image is not None and lines_field is not None:
		n_lines = lines_image.shape[0]
		
		for i in range(n_lines):
			a, b, c = lines_image[i]
			a_p, b_p, c_p = lines_field[i]
			A.append([a_p, b_p, c_p, 0, 0, 0, -a * a_p, -a * b_p, -a * c_p])
			A.append([0, 0, 0, a_p, b_p, c_p, -b * a_p, -b * b_p, -b * c_p])'''
	
	A = np.array(A)
	
	U, S, Vt = np.linalg.svd(A)
	
	h = Vt[-1]
	
	H = h.reshape(3, 3)
	
	return H #H / H[-1, -1]

def compute_homography_prepared(guessed_keypoints, guessed_lines):
	if len(guessed_keypoints) >= 4:# + len(guessed_lines) >= 4:
		points_image = None
		points_field = None
		
		lines_image = None
		lines_field = None
		
		keys = guessed_keypoints.keys()
		if len(guessed_keypoints) != 0:
			points_image = np.array([guessed_keypoints[key] for key in keys], dtype=np.float32)
			points_field = np.array([FIELD_KP_COORDS[key] for key in keys], dtype=np.float32)
		
		keys = guessed_lines.keys()
		#if len(guessed_lines) != 0:
			#lines_image = np.array([guessed_lines[key] for key in keys], dtype=np.float32)
			#lines_field = np.array([FIELD_KL_VALUES[key] for key in keys], dtype=np.float32)
		
		try:
			#my_h = compute_homography(points_image, points_field, lines_image, lines_field)
			
			H, _ = cv2.findHomography(points_field, points_image)
			
			#print(f'my_H: {my_h}\ncv_H: {H}')
			
			return H
		except np.linalg.LinAlgError: # Singular matrix, for example
			return None
	else:
		return None

def compute_inverse_homography(guessed_keypoints, guessed_lines):
	H = compute_homography_prepared(guessed_keypoints, guessed_lines)
	
	if H is not None:
		det = np.linalg.det(H)
		#print('det:', det)
		
		if det > -0.0001 or det < -1.0: # unrealistic/unstable/faulty matrix
			return None
		else:
			return np.linalg.inv(H)
	else:
		return None


KP_ANN_BOX_SIZE = 30 # in pixels


ENTITY_ATTACKER = 0
ENTITY_ATTACKER_HAS_BALL = 1
ENTITY_KEEPER = 2
ENTITY_ANOTHER_PERSON = 3
ENTITY_BALL = 4

class Entity:
	def __init__(self, cls, x, y):
		self.cls = cls # ENTITY_X
		self.x = x # normalize to [0.0, 1.0]
		self.y = y # normalize to [0.0, 1.0]

def process_frame_yolo(frame, prev_keypoints, main_model, keypoints_model, ball_model, classify_model, variant, process_players, process_ball):
	frame_width = frame.shape[1]
	frame_height = frame.shape[0]
	
	half_width = frame_width // 2
	half_height = frame_height // 2
	
	do_main = False
	do_ball = False
	do_keypoints = False
	do_classify = False
	
	if process_players:
		do_main = True
		do_keypoints = True
	
	if process_ball:
		do_ball = True
		do_classify = True
	
	if variant == VARIANT_PLAYERS:
		do_main = True
	elif variant == VARIANT_HIT:
		do_main = True
		do_classify = True
	elif variant == VARIANT_FIELD:
		do_main = True
	elif variant == VARIANT_KEYPOINTS:
		do_main = True
		do_keypoints = True
	
	
	
	result_main = None
	results_ball = None
	result_keypoints = None
	
	if do_main:
		result_main = main_model.predict(frame)[0]
	if do_ball:
		top_left = frame[:half_height, :half_width]
		top_right = frame[:half_height, half_width:]
		bottom_left = frame[half_height:, :half_width]
		bottom_right = frame[half_height:, half_width:]
		
		results_ball = [ball_model.predict(top_left)[0], ball_model.predict(top_right)[0], ball_model.predict(bottom_left)[0], ball_model.predict(bottom_right)[0]]
	if do_keypoints:
		result_keypoints = keypoints_model.predict(frame)[0]
	
	
	
	
	
	entities = [] #[Entity, ...]
	
	field_mask = None
	
	if result_main is not None: # TODO + those who has ball
		field_mask = make_field_mask(frame, result_main)
		
		boxes = result_main.boxes.xywh.cpu()
		
		for n, box in enumerate(boxes):
			yolo_cls = result_main.boxes.cls[n]
			
			if result_main.boxes.conf[n] >= CONFIDENCE_THRESHOLD:
				x, y, _, _ = box
				x /= frame_width
				y /= frame_height
				
				if yolo_cls == ATTACKER_CLASS:
					entities.append(Entity(ENTITY_ATTACKER, x, y))
				elif yolo_cls == KEEPER_CLASS:
					entities.append(Entity(ENTITY_KEEPER, x, y))
				elif yolo_cls == ANOTHER_PERSON_CLASS:
					entities.append(Entity(ENTITY_ANOTHER_PERSON, x, y))
	
	
	
	if result_keypoints is not None:
		guessed_keypoints = guessed_keypoints_from_result(result_keypoints, prev_keypoints, frame_width, frame_height)
		all_lines, guessed_lines = guess_lines(guessed_keypoints, cv2.bitwise_and(frame, field_mask))
		
		inv_H = compute_inverse_homography(guessed_keypoints, data_lines(guessed_lines))
	else:
		guessed_keypoints = None
		all_lines = None
		guessed_lines = None
		inv_H = None
	
	
	
	if results_ball is not None:
		for i, result in enumerate(results_ball):
			if i == 0: # top_left
				off_x = 0
				off_y = 0
			elif i == 1: # top_right
				off_x = half_width
				off_y = 0
			elif i == 2: # bottom_left
				off_x = 0
				off_y = half_height
			elif i == 3: # bottom_right
				off_x = half_width
				off_y = half_height
			
			boxes = result.boxes.xyxy.cpu()
			
			for (x0, y0, x1, y1) in boxes:
				px0 = int(x0 + off_x)
				py0 = int(y0 + off_y)
				px1 = int(x1 + off_x)
				py1 = int(y1 + off_y)
				
				entities.append(Entity(ENTITY_BALL, (px0 + px1)/2/frame_width, (py0 + py1)/2/frame_height))
	
	
	
	out_frame = None
	
	if variant == VARIANT_VIDEO:
		out_frame = frame
	elif variant == VARIANT_PLAYERS:
		out_frame = result_main.plot()
	elif variant == VARIANT_HIT:
		out_frame = process_hit_ball(frame, result_main, classify_model)
		
		if results_ball is not None:
			for i, result in enumerate(results_ball):
				if i == 0: # top_left
					off_x = 0
					off_y = 0
				elif i == 1: # top_right
					off_x = half_width
					off_y = 0
				elif i == 2: # bottom_left
					off_x = 0
					off_y = half_height
				elif i == 3: # bottom_right
					off_x = half_width
					off_y = half_height
				
				boxes = result.boxes.xyxy.cpu()
				
				for (x0, y0, x1, y1) in boxes:
					px0 = int(x0 + off_x)
					py0 = int(y0 + off_y)
					px1 = int(x1 + off_x)
					py1 = int(y1 + off_y)
					
					cv2.rectangle(out_frame, (px0, py0), (px1, py1), color=(0, 255, 255), thickness=2)
	
	elif variant == VARIANT_FIELD:
		out_frame = cv2.bitwise_and(frame, field_mask)
	elif variant == VARIANT_KEYPOINTS:
		out_frame = frame.copy()
		
		if guessed_keypoints is not None:
			for (kp, coords) in guessed_keypoints.items():
				label = kp_name(kp)
				color = kp_color(kp)[::-1]
				
				x0, y0, x1, y1 = (coords[0]*frame_width - KP_ANN_BOX_SIZE/2, coords[1]*frame_height - KP_ANN_BOX_SIZE/2, coords[0]*frame_width + KP_ANN_BOX_SIZE/2, coords[1]*frame_height + KP_ANN_BOX_SIZE/2)
				
				cv2.rectangle(out_frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
				cv2.putText(out_frame, label, (int(x0), int(y0) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
		
		'''if guessed_lines is not None:
			for (kl, values) in guessed_lines.items():
				color = kl_color(kl)[::-1]
				
				a, b, c = values
				
				if b != 0:
					x0 = 0
					y0 = -c / b
					
					x1 = 1.0
					y1 = -(a + c) / b
				else: 
					x0 = -c / a
					y0 = 0 
					
					x1 = -c / a
					y1 = 1.0
				
				cv2.line(out_frame, (int(x0 * frame_width), int(y0 * frame_height)), (int(x1 * frame_width), int(y1 * frame_height)), color, 2)'''
	
	if variant != None:
		out_frame = cv2.resize(out_frame, (640, 320), interpolation=cv2.INTER_LINEAR)
	
	return out_frame, field_mask, all_lines, guessed_keypoints, inv_H, entities


class YOLOProcessor(QObject):
	processed_data_signal = pyqtSignal(object)
	
	def __init__(self, main_model, keypoints_model, ball_model, classify_model):
		super().__init__()
		self.running = False
		
		self.main_model = main_model
		self.keypoints_model = keypoints_model
		self.ball_model = ball_model
		self.classify_model = classify_model
		
		self.variant = VARIANT_VIDEO
		
		self.process_players = True
		self.process_ball = False
	
	def set_variant(self, variant):
		self.variant = variant
	
	def set_process_players(self, value):
		self.process_players = value
	
	def set_process_ball(self, value):
		self.process_ball = value
	
	def start_processing(self, frame_queue):
		self.running = True
		while self.running:
			if not frame_queue.empty():
				in_data = frame_queue.get()
				frame, prev_keypoints = in_data
				processed_data = process_frame_yolo(frame, prev_keypoints, self.main_model, self.keypoints_model, self.ball_model, self.classify_model, self.variant, self.process_players, self.process_ball)
				self.processed_data_signal.emit(processed_data)

	def stop_processing(self):
		self.running = False



def draw_square_with_alpha(image, center, size, color, alpha):
	height, width, _ = image.shape
	
	x0 = center[0] - size//2
	y0 = center[1] - size//2
	x1 = center[0] + size//2
	y1 = center[1] + size//2
	
	if x0 >= 0 and x1 < width and y0 >= 0 and y1 < height:
		square = np.zeros((size, size, 3))
		
		square[:, :] = color
		
		region = image[y0 : y1, x0: x1]
		
		result = square * alpha + region * (1.0 - alpha)
		
		image[y0 : y1, x0 : x1] = result



class SoftwareWindow(QMainWindow):
	def __init__(self, ui_path, field_image_path, main_model_path, keypoints_model_path, ball_model_path, classify_model_path):
		super().__init__()
		self.setWindowTitle("Hockey Process")
		
		loadUi(ui_path, self)
		
		
		self.yolo_processor = YOLOProcessor(main_model=YOLO(main_model_path), keypoints_model=YOLO(keypoints_model_path), ball_model=YOLO(ball_model_path), classify_model=YOLO(classify_model_path))
		self.yolo_thread = None
		self.frame_queue = queue.Queue()
		
		self.processed_frame = None
		self.field_mask = None
		self.keypoints = None
		self.inv_H = None
		self.entities = None
		
		self.heatmap_data = []
		
		self.each_frame_value = 5
		
		self.out_video_writer = None
		
		
		
		self.video_label = self.findChild(QLabel, "video_label")
		
		self.set_video_button = self.findChild(QPushButton, "set_video")
		self.video_path_line_edit = self.findChild(QLineEdit, "video_path")
		
		self.each_frame = self.findChild(QLineEdit, "each_frame")
		self.each_frame.setText(str(self.each_frame_value))
		
		self.start_button = self.findChild(QPushButton, "start")
		self.stop_button = self.findChild(QPushButton, "stop")
		self.stop_button.setEnabled(False)
		
		self.button_gen_highlight_video = self.findChild(QPushButton, "gen_highlight_video")
		self.button_stop_highlight_video = self.findChild(QPushButton, "stop_highlight_video")
		self.button_stop_highlight_video.setEnabled(False)
		
		self.button_show_heatmap = self.findChild(QPushButton, "button_show_heatmap")
		self.button_show_heatmap.setEnabled(False)
		
		self.radio_none = self.findChild(QRadioButton, "radio_none")
		self.radio_video = self.findChild(QRadioButton, "radio_video")
		self.radio_players = self.findChild(QRadioButton, "radio_players")
		self.radio_hit = self.findChild(QRadioButton, "radio_hit")
		self.radio_field = self.findChild(QRadioButton, "radio_field")
		self.radio_keypoints = self.findChild(QRadioButton, "radio_keypoints")
		
		self.checkbox_players = self.findChild(QCheckBox, "checkbox_players")
		self.checkbox_ball = self.findChild(QCheckBox, "checkbox_ball")
		
		
		self.field_label = self.findChild(QLabel, "field_label")
		self.field_image = cv2.cvtColor(cv2.imread(field_image_path), cv2.COLOR_BGR2RGB)
		self.update_field()
		
		
		self.cap = None
		self.frame_index = 0
		
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_frame)
		self.timer.start(int(1000/FPS))
		
		
		self.set_video_button.clicked.connect(self.set_video_source)
		
		self.each_frame.textChanged.connect(self.set_each_frame_value)
		
		self.start_button.clicked.connect(self.start_yolo_thread)
		self.stop_button.clicked.connect(self.stop_yolo_thread)
		self.yolo_processor.processed_data_signal.connect(self.update_processed_data)
		
		self.button_show_heatmap.clicked.connect(self.show_heatmap)
		
		self.button_gen_highlight_video.clicked.connect(self.start_highlight_video)
		self.button_stop_highlight_video.clicked.connect(self.save_highlight_video)
		
		self.radio_none.toggled.connect(self.on_radio_button_toggled)
		self.radio_video.toggled.connect(self.on_radio_button_toggled)
		self.radio_players.toggled.connect(self.on_radio_button_toggled)
		self.radio_hit.toggled.connect(self.on_radio_button_toggled)
		self.radio_field.toggled.connect(self.on_radio_button_toggled)
		self.radio_keypoints.toggled.connect(self.on_radio_button_toggled)
		
		self.video_path_line_edit.textChanged.connect(self.update_video_source)
		
		self.checkbox_players.toggled.connect(self.on_checkbox_toggle)
		self.checkbox_ball.toggled.connect(self.on_checkbox_toggle)
	
	def start_highlight_video(self):
		files = os.listdir('highlights')
		
		i = 0
		name = f'highlight-{i}.mp4'
		while name in files:
			name = f'highlight-{i}.mp4'
			i += 1
		
		output_path = os.path.join('highlights', name)
		
		self.button_gen_highlight_video.setEnabled(False)
		self.button_stop_highlight_video.setEnabled(True)
		
		frame_rate = 30
		frame_width = 640
		frame_height = 320
		
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.out_video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
	
	def save_highlight_video(self):
		self.button_gen_highlight_video.setEnabled(True)
		self.button_stop_highlight_video.setEnabled(False)
		
		self.out_video_writer.release()
		
		self.out_video_writer = None
	
	def update_field(self):
		display_frame = self.field_image.copy()
		height, width, _ = display_frame.shape
		
		
		if self.field_mask is not None and self.inv_H is not None:
			x0, y0, w, h = cv2.boundingRect(self.field_mask[:, :, 0])
			x1 = (x0 + w)/self.field_mask.shape[1]
			y1 = (y0 + h)/self.field_mask.shape[0]
			x0 = x0/self.field_mask.shape[1]
			y0 = y0/self.field_mask.shape[0]
			
			left_bottom = np.array([x0, y1, 1.0], dtype=np.float32)
			right_bottom = np.array([x1, y1, 1.0], dtype=np.float32)
			left_top = np.array([x0, y0, 1.0], dtype=np.float32)
			right_top = np.array([x1, y0, 1.0], dtype=np.float32)
			
			field_lb = self.inv_H @ left_bottom
			field_lb /= field_lb[2]
			field_rb = self.inv_H @ right_bottom
			field_rb /= field_rb[2]
			field_lt = self.inv_H @ left_top
			field_lt /= field_lt[2]
			field_rt = self.inv_H @ right_top
			field_rt /= field_rt[2]
			
			pixel_coords = [
				(int(field_lb[0]*FIELD_IMAGE_SCALE + width/2), int(-field_lb[1]*FIELD_IMAGE_SCALE + height/2)),
				(int(field_rb[0]*FIELD_IMAGE_SCALE + width/2), int(-field_rb[1]*FIELD_IMAGE_SCALE + height/2)),
				(int(field_rt[0]*FIELD_IMAGE_SCALE + width/2), int(-field_rt[1]*FIELD_IMAGE_SCALE + height/2)),
				(int(field_lt[0]*FIELD_IMAGE_SCALE + width/2), int(-field_lt[1]*FIELD_IMAGE_SCALE + height/2)),
			]
			
			pixel_coords = np.array([pixel_coords], dtype=np.int32)
			
			camera_fill_color = (255, 0, 255)
			alpha = 0.35
			
			overlay = np.zeros(display_frame.shape, dtype=np.uint8)
			cv2.fillPoly(overlay, pixel_coords, camera_fill_color)
			
			cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
		
		
		thickness = 2
		color = (0, 0, 0)
		
		if self.inv_H is not None:
			for entity in self.entities:
				if entity.cls == ENTITY_ATTACKER:
					color = (255, 0, 0) # red
					radius = 10
				elif entity.cls == ENTITY_ATTACKER_HAS_BALL:
					color = (255, 0, 255) # purple
					radius = 10
				elif entity.cls == ENTITY_KEEPER:
					color = (255, 255, 255) # white
					radius = 10
				elif entity.cls == ENTITY_ANOTHER_PERSON:
					radius = 10
					color = (255, 255, 0) # yellow
				elif entity.cls == ENTITY_BALL:
					color = (80, 80, 80) # grey
					radius = 5
				
				image_coord = np.array([entity.x, entity.y, 1], dtype=np.float32)
				
				field_coord = self.inv_H @ image_coord
				field_coord /= field_coord[2] # normalization
				
				entity_x = field_coord[0]
				entity_y = field_coord[1]
				
				px = int(entity_x*FIELD_IMAGE_SCALE + width/2)
				py = int((-entity_y*FIELD_IMAGE_SCALE) + height/2)
				
				cv2.circle(display_frame, (px, py), radius, color, thickness)
		
		
		display_frame = cv2.resize(display_frame, (width//2, height//2))
		height, width, _ = display_frame.shape
		bytes_per_line = 3 * width
		
		q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
		
		self.field_label.setPixmap(QPixmap.fromImage(q_image))
	
	def update_heatmap_data(self):
		for entity in self.entities:
			if entity.cls == ENTITY_ATTACKER or entity.cls == ENTITY_ATTACKER_HAS_BALL:
				image_coord = np.array([entity.x, entity.y, 1], dtype=np.float32)
				
				field_coord = self.inv_H @ image_coord
				field_coord /= field_coord[2] # normalization
				
				self.heatmap_data.append(field_coord[:2])
		
		if len(self.heatmap_data) != 0:
			self.button_show_heatmap.setEnabled(True)
	
	def show_heatmap(self):
		if len(self.heatmap_data) != 0:
			display_frame = self.field_image.copy()
			height, width, _ = display_frame.shape
			
			quad_size = 8
			quad_color = (255, 0, 0)
			
			for x, y in self.heatmap_data:
				px = int(x*FIELD_IMAGE_SCALE + width/2)
				py = int((-y*FIELD_IMAGE_SCALE) + height/2)
				
				draw_square_with_alpha(display_frame, (px, py), quad_size, quad_color, 0.02)
			
			fig, ax = plt.subplots()
			ax.imshow(display_frame)
			
			ax.axis('off')
			plt.show()
	
	def set_each_frame_value(self, text):
		n = self.each_frame_value
		try:
			n = int(text)
		except:
			pass
		
		if n != 0:
			self.each_frame_value = n
	
	def set_video_source(self):
		video_source, _ = QFileDialog.getOpenFileName(self, "Select a video file", "", "")
		if video_source:
			self.video_path_line_edit.setText(video_source)
			
			self.cap = cv2.VideoCapture(video_source)
			self.frame_index = 0
	
	def update_video_source(self, video_source):
		try:
			cap = cv2.VideoCapture(video_source)
			
			self.cap = cap
			self.frame_index = 0
		except:
			pass
	
	def on_radio_button_toggled(self):
		selected_button = self.sender()
		if selected_button.isChecked():
			if selected_button.objectName() == "radio_none":
				self.yolo_processor.set_variant(None)
			if selected_button.objectName() == "radio_video":
				self.yolo_processor.set_variant(VARIANT_VIDEO)
			elif selected_button.objectName() == "radio_players":
				self.yolo_processor.set_variant(VARIANT_PLAYERS)
			elif selected_button.objectName() == "radio_hit":
				self.yolo_processor.set_variant(VARIANT_HIT)
			elif selected_button.objectName() == "radio_field":
				self.yolo_processor.set_variant(VARIANT_FIELD)
			elif selected_button.objectName() == "radio_keypoints":
				self.yolo_processor.set_variant(VARIANT_KEYPOINTS)
	
	def on_checkbox_toggle(self):
		selected_checkbox = self.sender()
		value = selected_checkbox.isChecked()
		
		if selected_checkbox.objectName() == "checkbox_players":
			self.yolo_processor.set_process_players(value)
		if selected_checkbox.objectName() == "checkbox_ball":
			self.yolo_processor.set_process_ball(value)
	
	def read_capture(self):
		if self.yolo_thread and self.yolo_processor.running:
			success, frame = self.cap.read()
			if success:
				if self.frame_index%self.each_frame_value == 0:
					kp = None
					if self.keypoints is not None:
						kp = self.keypoints.copy()
					
					self.frame_queue.put((frame, kp))
				
				self.frame_index += 1
			else:
				self.cap.release()
				self.cap = None
	
	def update_frame(self):
		if self.cap is not None:
			if self.processed_frame is not None:
				display_frame = self.processed_frame
				
				if self.out_video_writer is not None:
					self.out_video_writer.write(display_frame)
				
				display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
				height, width, _ = display_frame.shape
				bytes_per_line = 3 * width
				
				q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
				
				self.video_label.setPixmap(QPixmap.fromImage(q_image))
				
				self.processed_frame = None
				
				self.read_capture()
			else:
				if self.frame_queue.empty():
					self.read_capture()
			
			if self.inv_H is not None:
				self.update_field()
				self.update_heatmap_data()
	
	def update_processed_data(self, data):
		processed_frame, field_mask, all_lines, keypoints, inv_H, entities = data
		
		self.processed_frame = processed_frame
		self.field_mask = field_mask
		self.all_lines = all_lines
		self.keypoints = keypoints
		self.inv_H = inv_H
		self.entities = entities
	
	def start_yolo_thread(self):
		if self.yolo_thread is None:
			self.yolo_thread = threading.Thread(target=self.yolo_processor.start_processing, args=(self.frame_queue,))
			self.yolo_thread.start()
		self.start_button.setEnabled(False)
		self.stop_button.setEnabled(True)
	
	def stop_yolo_thread(self):
		if self.yolo_processor.running:
			self.yolo_processor.stop_processing()
			self.yolo_thread.join()
			self.yolo_thread = None
		self.start_button.setEnabled(True)
		self.stop_button.setEnabled(False)
	
	def closeEvent(self, event):
		self.stop_yolo_thread()
		if self.cap is not None:
			self.cap.release()
		super().closeEvent(event)


if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = SoftwareWindow(UI_FILE_PATH, HOCKEY_FIELD_IMAGE_PATH, MAIN_MODEL_PATH, KEYPOINTS_MODEL_PATH, BALL_MODEL_PATH, CLASSIFY_MODEL_PATH)
	window.resize(680, 600)
	window.show()
	sys.exit(app.exec())
