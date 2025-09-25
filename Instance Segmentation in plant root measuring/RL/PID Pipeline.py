# --- IMPORTS ---
import os
import cv2
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from patchify import patchify, unpatchify
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from pid_controller import PIDController
from sim_class import Simulation

# --- METRIC ---
def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return TP / (Positives + K.epsilon())

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return TP / (Pred_Positives + K.epsilon())

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2.0 * ((precision * recall) / (precision + recall + K.epsilon()))

# --- IMAGE CROP ---
def trim_picture(picture):
    picture = picture[:, :-150, :]
    xo, yo, _ = picture.shape
    x_mid = xo // 2
    valid_columns = (picture[x_mid, :, :] >= 100).any(axis=1)
    col_indices = np.where(valid_columns)[0]
    if len(col_indices) == 0:
        return picture, (0, 0), 0
    left_in_cut = col_indices[0]
    trimmed_picture1 = picture[:, col_indices, :]

    y_mid = trimmed_picture1.shape[1] // 2
    valid_rows = (trimmed_picture1[:, y_mid, :] >= 100).any(axis=1)
    row_indices = np.where(valid_rows)[0]
    if len(row_indices) == 0:
        return trimmed_picture1, (0, 0), left_in_cut
    top_in_cut = row_indices[0]
    trimmed_picture2 = trimmed_picture1[row_indices, :, :]

    x, y, _ = trimmed_picture2.shape
    min_dim = min(x, y)
    top_in_trimmed = (x - min_dim) // 2
    left_in_trimmed = (y - min_dim) // 2

    final_crop = trimmed_picture2[
        top_in_trimmed : top_in_trimmed + min_dim,
        left_in_trimmed : left_in_trimmed + min_dim,
        :
    ]

    offset_x_full = top_in_cut + top_in_trimmed
    offset_y_full = left_in_cut + left_in_trimmed

    return final_crop, (offset_x_full, offset_y_full), left_in_cut

# --- IMAGE PADDING ---
def padder(image, patch_size):
    h, w = image.shape[:2]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w
    top, bottom = height_padding // 2, height_padding - height_padding // 2
    left, right = width_padding // 2, width_padding - width_padding // 2
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded, (top, bottom, left, right)

# --- FULL IMAGE PROCESSING ---
def advanced_image_processing(image_path, model, patch_size=256):
    image_color = cv2.imread(image_path)
    if image_color is None:
        raise ValueError(f"Could not load image at {image_path}")

    image_trimmed, (offset_x, offset_y), left_in_cut = trim_picture(image_color)
    image_gray = cv2.cvtColor(image_trimmed, cv2.COLOR_BGR2GRAY)
    trimmed_shape = image_gray.shape[:2]

    padded, padding_info = padder(image_gray, patch_size)
    padded = np.expand_dims(padded, axis=-1)
    patches_ = patchify(padded, (patch_size, patch_size, 1), step=patch_size)
    reshaped = patches_.reshape(-1, patch_size, patch_size, 1) / 255.0

    predicted = model.predict(reshaped)
    num = int(math.sqrt(predicted.shape[0]))
    predicted_reshaped = predicted.reshape(num, num, patch_size, patch_size)
    mask = unpatchify(predicted_reshaped, padded.shape[:2])
    mask = (mask * 255).astype(np.uint8)
    mask[mask >= 128] = 255
    mask[mask < 128] = 0

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    t, b, l, r = padding_info
    corrected = mask[t:-b or None, l:-r or None]

    final_mask = np.zeros(trimmed_shape, dtype=np.uint8)
    x_off = (trimmed_shape[0] - corrected.shape[0]) // 2
    y_off = (trimmed_shape[1] - corrected.shape[1]) // 2
    final_mask[x_off:x_off + corrected.shape[0], y_off:y_off + corrected.shape[1]] = corrected

    return final_mask, (offset_x, offset_y), left_in_cut

# --- FIND ROOT TIPS ---
def extract_endpoints(mask, n=5, min_area=230):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [
        (x, y, w, h, cv2.contourArea(c))
        for c in contours if cv2.contourArea(c) >= min_area
        for (x, y, w, h) in [cv2.boundingRect(c)]
    ]
    bboxes = sorted(bboxes, key=lambda b: b[4], reverse=True)[:n]

    centroids = []

    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray')

    for (x, y, w, h, _) in bboxes:
        roi = mask[y:y + h, x:x + w]
        M = cv2.moments(roi)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            white_points = np.argwhere(roi == 255)
            if white_points.any():
                idx = np.argmax(white_points[:, 0])
                end_y, end_x = white_points[idx]
                global_x = x + end_x
                global_y = y + end_y
                centroids.append((global_x, global_y))

                ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none'))
                ax.plot(global_x, global_y, 'bo', markersize=5)

    plt.title("Predicted Mask + Root Tips")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return centroids

# --- CONVERSIONS ---
def pixel_to_mm(coords_px, original_image_shape, plate_size_mm=150, offset_px_x=0):
    height_px = original_image_shape[0]
    factor = plate_size_mm / height_px
    offset_mm_x = offset_px_x * factor - 5  # adjusting the location because of previous cropping
    return [(x * factor - offset_mm_x, y * factor) for x, y in coords_px]

def mm_to_robot_coords(coords_mm, origin=np.array([0.10775, 0.088 - 0.026, 0.1695])):
    return [[origin[0] + (y / 1000.0), origin[1] + (x / 1000.0), origin[2]] for x, y in coords_mm]

def apply_min_velocity(signal, min_val=0.003):
    return [s if abs(s) > min_val else 0.0 for s in signal]

# --- MAIN LOOP ---
if __name__ == "__main__":
    model = load_model("../datalab_tasks/task5/saved_modelr4.h5", custom_objects={"f1": f1})
    sim = Simulation(num_agents=1, render=True)
    sim.reset(num_agents=1)
    time.sleep(1)

    image_path = sim.get_plate_image()
    full_image = cv2.imread(image_path)
    if full_image is None:
        raise ValueError("Failed to read the sim image.")
    full_shape = full_image.shape[:2]

    # Get mask + tips
    mask, (offset_x, offset_y), left_in_cut = advanced_image_processing(image_path, model)
    centroids_px = extract_endpoints(mask, n=5)
    if not centroids_px:
        exit(1)

    # Convert to robot space
    centroids_px_global = [(x + offset_y, y + offset_x) for (x, y) in centroids_px]
    centroids_mm = pixel_to_mm(centroids_px_global, full_shape, plate_size_mm=150, offset_px_x=left_in_cut)
    targets = mm_to_robot_coords(centroids_mm)

    # Control loop
    for i, target in enumerate(targets):
        print(f"[INFO] Moving to root tip {i+1} => {target}")
        pid_x = PIDController(3.0, 0.01, 0.4)
        pid_y = PIDController(3.0, 0.01, 0.4)
        pid_z = PIDController(3.0, 0.01, 0.4)

        reached = False
        step = 0
        max_steps = 1500

        while not reached and step < max_steps:
            state = sim.run([[0, 0, 0, 0]], num_steps=1)
            robot_id = next(iter(state))
            pos = np.array(state[robot_id]["pipette_position"])
            err_xyz = np.array(target) - pos

            signal = [
                pid_x.update(target[0], pos[0], dt=1),
                pid_y.update(target[1], pos[1], dt=1),
                pid_z.update(target[2], pos[2], dt=1),
                0
            ]
            signal = apply_min_velocity(signal)
            signal = np.clip(signal, -0.3, 0.3)
            sim.run([signal], num_steps=1)

            if step % 100 == 0:
                print(f"Step {step} | Error norm: {np.linalg.norm(err_xyz):.6f}")

            if np.linalg.norm(err_xyz) < 0.001 or step == max_steps - 1:
                print(f"[✓] Inoculating at step {step} | error {np.linalg.norm(err_xyz):.6f}")
                for _ in range(5):
                    sim.run([[0, 0, 0, 0]], num_steps=1)
                sim.run([[0, 0, 0, 1]], num_steps=1)
                for _ in range(5):
                    sim.run([[0, 0, 0, 0]], num_steps=1)
                if i == len(targets) - 1:
                    for _ in range(10 * 5):
                        sim.run([[0, 0, 0, 0]], num_steps=1)
                reached = True

            step += 1
