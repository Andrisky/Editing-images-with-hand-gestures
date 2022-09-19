import cv2
import numpy as np
import os
from math import sqrt, acos, atan2

# Variables and lists
hand_features = False

editing = False
save = False
delete = False
initial_len = 0

enabled_gestures = True

scroll_images = True
view_images = True
work_copies = []
image_pointer = 0
equals = False

operator_selected = False
scroll_operators = False
operator_pointer = 0

box_placed = False
box_shifting = False
crop = False
pinch = False
tilt = 20
# 0 for the first implementation of the pinch gestures
# 1 for the second one
switch = 1

operators = ['Crop image',
             'Exponential operator',
             'Gaussian filter', 'Bilateral filter', 'Median filter',
             'Dilation', 'Erosion']

edited_work_copy = None

map_dis_r = {i: round((0.4 + (14 - i) * 0.15), 2) for i in range(14, 4, -1)}
dict_kernel = {i + 4: 2 * i - 1 for i in range(1, 11, 1)}
dict_bar = {i: (14 - i + 1) * 20 for i in range(14, 4, -1)}


# trackbars
# def posTrackbar(pos):
#     global H_min, S_min, V_min, H_max, S_max, V_max
#
#     H_min = cv2.getTrackbarPos('H_min', 'hsv')
#     S_min = cv2.getTrackbarPos('S_min', 'hsv')
#     V_min = cv2.getTrackbarPos('V_min', 'hsv')
#
#     H_max = cv2.getTrackbarPos('H_max', 'hsv')
#     S_max = cv2.getTrackbarPos('S_max', 'hsv')
#     V_max = cv2.getTrackbarPos('V_max', 'hsv')
#
#
# hsv = cv2.namedWindow('hsv')
#
# cv2.createTrackbar('H_min', 'hsv', 0, 255, posTrackbar)
# cv2.createTrackbar('S_min', 'hsv', 0, 255, posTrackbar)
# cv2.createTrackbar('V_min', 'hsv', 0, 255, posTrackbar)
#
# cv2.createTrackbar('H_max', 'hsv', 0, 255, posTrackbar)
# cv2.createTrackbar('S_max', 'hsv', 0, 255, posTrackbar)
# cv2.createTrackbar('V_max', 'hsv', 0, 255, posTrackbar)

H_min = 25
S_min = 52
V_min = 52
H_max = 130
S_max = 255
V_max = 255

if __name__ == '__main__':

    dir_path = 'test images'
    # Change the current working directory to the path containing the test images
    os.chdir(dir_path)
    image_names = os.listdir(os.getcwd())

    # Read the images and store it in a list called images
    images = []
    for i in range(len(image_names)):
        image = cv2.imread(image_names[i])
        images.append(image)

        if 'Saved_image' not in image_names[i]:
            initial_len += 1

    if len(images) == 0:
        print('\n\u001b[31m'
              '------------------------------------------------------------------')
        print('|| WARNING:                                                     ||'
              '\n|| The directory test images is currently empty.                ||'
              '\n|| Insert images into the directory before starting the system. ||')
        print('------------------------------------------------------------------')
        exit(0)

    # Create a VideoCapture object)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Flip the frame
        frame = cv2.flip(frame, 1)
        # Create a space for entering status information
        caption = np.zeros((90, frame.shape[1], frame.shape[2]), np.uint8)
        frame = np.concatenate((frame, caption), axis=0)

        # Recommended Hand position
        # cv2.rectangle(frame, (490, 130), (570, 290), (0, 255, 255), 1)

        # Make a ROI on the flipped frame
        cv2.rectangle(frame, (430, 90), (630, 290), (0, 255, 0), 1)
        roi = frame[91:290, 431:630]

        # Segmentation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([H_min, S_min, V_min])
        upper_bound = np.array([H_max, S_max, V_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        # Median filtering is an effective way to reducing impulse noise
        mask = cv2.medianBlur(mask, 9)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Contour and convex hull
        contour, _ = cv2.findContours(mask, None, cv2.CHAIN_APPROX_SIMPLE)
        res = np.zeros(roi.shape, np.uint8)
        try:
            contour = max(contour, key=lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(contour)

            area_contour = cv2.contourArea(contour)
            area_hull = cv2.contourArea(hull)
            area_ratio = ((area_hull - area_contour) / area_hull) * 100

            # Draw hull and contour
            contour_frame, hull_frame = contour + [430, 90], hull + [430, 90]
            cv2.drawContours(frame, [contour_frame], -1, [255, 255, 255], 2)

            # Find convexity defects of the hand
            hull = cv2.convexHull(contour, returnPoints=False)
            convexity_defects = cv2.convexityDefects(contour, hull)
            defects = 0
            gesture_defects = 0
            for i in range(convexity_defects.shape[0]):
                s, e, f, d = convexity_defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                # finding the angle of the defect using cosine law
                angle = (acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                if angle <= 130 and a > 30 and c > 30 and b > 30:
                    cv2.circle(res, far, 5, (0, 255, 255), -1)
                    gesture_defects += 1

                if angle <= 95:
                    cv2.circle(res, far, 5, (0, 255, 0), -1)
                    defects += 1

                alpha = end[1] - start[1]
                beta = start[0] - end[0]
                gamma = end[0] * start[1] - start[0] * end[1]
                distance_point_line = (alpha * far[0] + beta * far[1] + gamma) / np.sqrt(alpha ** 2 + beta ** 2)

                if editing:
                    if distance_point_line > 30 and gesture_defects == 1 and switch == 1:
                        # draw on frame
                        if pinch:
                            cv2.line(frame, (start[0] + 430, start[1] + 90), (end[0] + 430, end[1] + 90), (0, 255, 0),
                                     4)
                            cv2.circle(frame, (start[0] + 430, start[1] + 90), 6, [0, 255, 0], -1)
                            cv2.circle(frame, (end[0] + 430, end[1] + 90), 6, [0, 255, 0], -1)

                if distance_point_line > 30 and gesture_defects == 1:
                    distance_start_end = int(np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) / 10) + 2

            # hand rotation:
            if defects == 0 and area_ratio < 10 and area_contour > 8000:
                n = len(contour)
                data_pts = np.empty((n, 2))

                for i in range(data_pts.shape[0]):
                    data_pts[i, 0] = contour[i, 0, 0]
                    data_pts[i, 1] = contour[i, 0, 1]
                # Perform PCA analysis
                _, eigenvectors, _ = cv2.PCACompute2(data_pts, None)

                hand_angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
                hand_angle = int(np.rad2deg(hand_angle) - 90)

                cv2.putText(res, str(hand_angle), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                if not editing:
                    cv2.line(frame, (430 + 145, 90 + 70), (430 + 165, 90 + 70), (0, 255, 0), 2)
                    cv2.line(frame, (430 + 155, 90 + 60), (430 + 155, 90 + 80), (0, 255, 0), 2)

                    cv2.line(frame, (430 + 54, 90 + 70), (430 + 34, 90 + 70), (0, 255, 0), 2)

            hand_features = True
        except:
            hand_features = False

        if hand_features:

            # The image to be modified has still to be selected
            if len(work_copies) == 0:
                # Show the current image
                if images[image_pointer] is not None:
                    cv2.imshow('Images', images[image_pointer])
                    view_images = True

                # Image scrolling
                if defects == 0 and area_ratio < 10 and area_contour > 8000:
                    delete = True
                    enabled_gestures = True
                    if -tilt <= hand_angle <= tilt and not scroll_images:
                        scroll_images = True
                    if hand_angle > tilt and scroll_images:
                        if image_pointer == len(images) - 1:
                            image_pointer = 0
                        else:
                            image_pointer += 1
                        scroll_images = False
                    elif hand_angle < -tilt and scroll_images:
                        if image_pointer == 0:
                            image_pointer = len(images) - 1
                        else:
                            image_pointer -= 1
                        scroll_images = False

                    # color
                    if hand_angle < -tilt and not scroll_images:
                        cv2.line(frame, (430 + 54, 90 + 70), (430 + 34, 90 + 70), (0, 0, 255), 2)
                    elif hand_angle > tilt and not scroll_images:
                        cv2.line(frame, (430 + 145, 90 + 70), (430 + 165, 90 + 70), (0, 0, 255), 2)
                        cv2.line(frame, (430 + 155, 90 + 60), (430 + 155, 90 + 80), (0, 0, 255), 2)

                # Image selection
                if defects == 4 and enabled_gestures:
                    work_copy = np.copy(images[image_pointer])
                    work_copies.append(work_copy)
                    enabled_gestures = False

                # Delete image
                if defects == 0 and area_ratio > 10 and delete:
                    scroll_images = False
                    top = tuple(contour[contour[:, :, 1].argmin()][0])

                    top_frame = [top[0] + 430, top[1] + 90]
                    cv2.circle(frame, top_frame, 6, [0, 0, 255], -1)
                    cv2.rectangle(frame, (430 + 110, 90 + 10), (430 + 195, 90 + 40), [255, 255, 255], 2)
                    cv2.putText(frame, 'delete', (430 + 115, 90 + 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    if 110 < top[0] < 195 and 10 < top[1] < 40:
                        cv2.rectangle(frame, (430 + 110, 90 + 10), (430 + 195, 90 + 40), [0, 0, 255], 2)
                        cv2.putText(frame, 'delete', (430 + 115, 90 + 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)

                        # Delete current image
                        image_name = image_names[image_pointer]
                        os.remove(image_name)
                        # Reread the images in the input image directory
                        images = []
                        image_names = os.listdir(os.getcwd())
                        if len(image_names) > 0:
                            for i in range(len(image_names)):
                                img = cv2.imread(image_names[i])
                                images.append(img)
                            image_pointer = len(images) - 1
                            delete = False
                        else:
                            print('\n\u001b[31m'
                                  '------------------------------------------------------------------')
                            print('|| WARNING:                                                     ||'
                                  '\n|| The directory test images is currently empty.                ||'
                                  '\n|| Insert images into the directory before starting the system. ||')
                            print('------------------------------------------------------------------')
                            exit(0)

                text = 'Images scrolling: image ' + str(image_pointer)  # + str(image_names[image_pointer])
                cv2.putText(frame, text, (30, 515), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # The image has been selected, but the operator has still to be selected
            else:  # elif not operator_selected:
                # Show a copy of the selected image
                cv2.imshow('Work copy', work_copies[-1])
                if view_images:
                    cv2.destroyWindow('Images')
                    view_images = False

                # Return to scroll images
                if defects == 0 and area_ratio < 10 and area_contour < 7000 and enabled_gestures and not operator_selected:
                    cv2.destroyWindow('Work copy')
                    work_copies = []
                    enabled_gestures = False

                # Save the current image (work copy)
                if len(work_copies) > 0:
                    equals = np.array_equal(work_copies[0], images[image_pointer])
                if defects == 0 and area_ratio > 10 and not equals and not operator_selected and save:
                    top = tuple(contour[contour[:, :, 1].argmin()][0])
                    top_frame = [top[0] + 430, top[1] + 90]
                    cv2.circle(frame, top_frame, 6, [0, 255, 0], -1)
                    cv2.rectangle(frame, (430 + 119, 90 + 10), (430 + 189, 90 + 40), [255, 255, 255], 2)
                    cv2.putText(frame, 'save', (430 + 125, 90 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    if 119 < top[0] < 189 and 10 < top[1] < 40:
                        cv2.rectangle(frame, (430 + 119, 90 + 10), (430 + 189, 90 + 40), [0, 255, 0], 2)
                        cv2.putText(frame, 'save', (430 + 125, 90 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)

                        # Save the edited image
                        image_name = 'Saved_image' + str(len(images) - initial_len) + '.jpg'
                        cv2.imwrite(image_name, work_copies[1])

                        # Reread the images in the input image directory
                        images = []
                        image_names = os.listdir(os.getcwd())
                        for i in range(len(image_names)):
                            image = cv2.imread(image_names[i])
                            images.append(image)

                        # Update flag values and the contents of the wrk_copies list
                        work_copies = []
                        scroll_images = False
                        save = False
                        delete = False
                        # Destroy the window named 'Work copy,
                        cv2.destroyWindow('Work copy')

                        image_pointer = len(images) - 1

                # Show which image has been selected
                text = 'image ' + str(image_pointer) + ' selected'
                cv2.putText(frame, text, (30, 515), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

                if not operator_selected:
                    text = 'Operators scrolling: ' + operators[operator_pointer]
                    cv2.putText(frame, text, (30, 550), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                    # Scrolling operators
                    if defects == 0 and area_ratio < 10 and area_contour > 8000 and not editing:
                        enabled_gestures = True
                        save = True

                        if -tilt <= hand_angle <= tilt and not scroll_operators:
                            scroll_operators = True
                        if hand_angle > tilt and scroll_operators:
                            if operator_pointer == len(operators) - 1:
                                operator_pointer = 0
                            else:
                                operator_pointer += 1
                            scroll_operators = False
                        elif hand_angle < -tilt and scroll_operators:
                            if operator_pointer == 0:
                                operator_pointer = len(operators) - 1
                            else:
                                operator_pointer -= 1
                            scroll_operators = False

                        # color
                        if hand_angle < -tilt and not scroll_operators:
                            cv2.line(frame, (430 + 54, 90 + 70), (430 + 34, 90 + 70), (0, 0, 255), 2)
                        elif hand_angle > tilt and not scroll_operators:
                            cv2.line(frame, (430 + 145, 90 + 70), (430 + 165, 90 + 70), (0, 0, 255), 2)
                            cv2.line(frame, (430 + 155, 90 + 60), (430 + 155, 90 + 80), (0, 0, 255), 2)

                    # Operator selection
                    if defects == 4 and not operator_selected and enabled_gestures:
                        operator_selected = True
                        enabled_gestures = False
                        work_copy = np.copy(work_copies[0])

            # Both the image and the operator has been selected
            if operator_selected:
                # Show which operator has been selected
                text = operators[operator_pointer] + ' selected'
                cv2.putText(frame, text, (30, 550), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)

                # Return on scroll operators
                if defects == 0 and area_ratio < 10 and area_contour < 7000:
                    if len(work_copies) == 2:
                        del work_copies[-1]

                    operator_selected = False
                    editing = False
                    enabled_gestures = False
                    box_placed = False
                    box_shifting = False
                    crop = False

                # Editing operator
                if operators[operator_pointer] == 'Crop image':
                    if gesture_defects == 1:
                        editing = True
                        if not box_placed:
                            pinch = True
                    if editing:
                        work_copy = np.copy(work_copies[0])
                        if not box_placed:
                            a = tuple(contour[contour[:, :, 1].argmin()][0])
                            b = tuple(contour[contour[:, :, 0].argmin()][0])
                            distance_extreme = int(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10)
                            if switch == 0:
                                dis = distance_extreme
                            elif switch == 1:
                                dis = distance_start_end

                            cv2.rectangle(frame, (20, 20), (60, 200), (1, 50, 32), -1)
                            cv2.rectangle(frame, (20, 20), (60, 200), (255, 255, 255), 2)
                            for i in range(8):
                                hs = 39 + i * 20
                                hf = hs + 2
                                cv2.rectangle(frame, (20, hs), (60, hf), (255, 255, 255), -1)

                            # draw
                            if gesture_defects == 1:
                                if switch == 0:
                                    cv2.line(frame, (a[0] + 430, a[1] + 90), (b[0] + 430, b[1] + 90), (0, 255, 0), 4)
                                    cv2.circle(frame, (a[0] + 430, a[1] + 90), 6, (0, 255, 0), -1)
                                    cv2.circle(frame, (b[0] + 430, b[1] + 90), 6, (0, 255, 0), -1)

                            h, w = work_copy.shape[0], work_copy.shape[1]
                            min_side = min(h, w)

                            if 5 <= dis <= 14:
                                l = int((dis * min_side / 25) + 150)
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                l = int((5 * min_side / 25) + 150)
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                l = int((15 * min_side / 25) + 150)
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            if l > min_side:
                                l = min_side

                            centre = (int(w / 2), int(h / 2))
                            ws = int(centre[0] - l / 2)
                            hs = int(centre[1] - l / 2)
                            wf = int(centre[0] + l / 2)
                            hf = int(centre[1] + l / 2)

                            edited_work_copy = cv2.rectangle(work_copy, (ws, hs), (wf, hf), [0, 255, 0], 2)

                            if len(work_copies) == 1:
                                work_copies.append(edited_work_copy)
                            else:
                                work_copies[1] = edited_work_copy

                        if gesture_defects == 2:
                            pinch = False
                            box_placed = True
                            box_shifting = True

                        if box_shifting and defects == 0 and area_ratio > 5:
                            crop = True
                            top = tuple(contour[contour[:, :, 1].argmin()][0])
                            cv2.circle(frame, (top[0] + 430, top[1] + 90), 6, [0, 255, 255], -1)

                            x, y = top[0], top[1]
                            # Normalization
                            norm_w = int(w / roi.shape[0])
                            norm_h = int(h / roi.shape[1])
                            # Saturation
                            ws = x + x * norm_w
                            if ws + l > w:
                                ws = w - l - 2
                            wf = l + ws
                            hs = y + y * norm_h
                            if hs + l > h:
                                hs = h - l - 2
                            hf = l + hs

                            edited_work_copy = cv2.rectangle(work_copy, (ws, hs), (wf, hf), [0, 255, 0], 2)

                            if len(work_copies) == 1:
                                work_copies.append(edited_work_copy)
                            else:
                                work_copies[1] = edited_work_copy

                        if defects == 1 and crop:
                            edited_work_copy = edited_work_copy[hs + 2:hf - 2, ws + 2:wf - 2]
                            if len(work_copies) == 1:
                                work_copies.append(edited_work_copy)
                            else:
                                work_copies[1] = edited_work_copy

                            work_copies[0] = edited_work_copy
                            box_shifting = False
                            editing = False
                            operator_selected = False
                            box_placed = False
                            crop = False

                # Operators with common gestures
                else:
                    if gesture_defects == 1:
                        editing = True
                        pinch = True
                    if editing:
                        work_copy = np.copy(work_copies[0])
                        a = tuple(contour[contour[:, :, 1].argmin()][0])
                        b = tuple(contour[contour[:, :, 0].argmin()][0])
                        distance_extreme = int(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10)

                        if switch == 0:
                            dis = distance_extreme
                        elif switch == 1:
                            dis = distance_start_end

                        cv2.rectangle(frame, (20, 20), (60, 200), (1, 50, 32), -1)
                        cv2.rectangle(frame, (20, 20), (60, 200), (255, 255, 255), 2)
                        for i in range(8):
                            hs = 39 + i * 20
                            hf = hs + 2
                            cv2.rectangle(frame, (20, hs), (60, hf), (255, 255, 255), -1)

                        if operators[operator_pointer] == 'Gaussian filter':
                            if 5 <= dis <= 14:
                                k = dict_kernel[dis]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                k = dict_kernel[5]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                k = dict_kernel[14]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            edited_work_copy = cv2.GaussianBlur(work_copy, (k, k), 0)

                        elif operators[operator_pointer] == 'Exponential operator':
                            if 5 <= dis <= 14:
                                r = map_dis_r[dis]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                r = map_dis_r[5]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                r = map_dis_r[14]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            edited_work_copy = np.asarray(((work_copy / 255) ** r) * 255, np.uint8)

                        elif operators[operator_pointer] == 'Bilateral filter':
                            if 5 <= dis <= 14:
                                k = dict_kernel[dis]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                k = dict_kernel[5]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                k = dict_kernel[14]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            edited_work_copy = cv2.bilateralFilter(work_copy, k, 75, 75)

                        elif operators[operator_pointer] == 'Median filter':
                            if 5 <= dis <= 14:
                                k = dict_kernel[dis]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                k = dict_kernel[5]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                k = dict_kernel[14]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            edited_work_copy = cv2.medianBlur(work_copy, k)

                        elif operators[operator_pointer] == 'Dilation':
                            if 5 <= dis <= 14:
                                k = dict_kernel[dis]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                k = dict_kernel[5]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                k = dict_kernel[14]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            kernel = np.ones((k, k), np.uint8)
                            edited_work_copy = cv2.dilate(work_copy, kernel, iterations=1)

                        elif operators[operator_pointer] == 'Erosion':
                            if 5 <= dis <= 14:
                                k = dict_kernel[dis]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[dis]), (255, 255, 255), -1)
                            elif dis < 5:
                                k = dict_kernel[5]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[5]), (255, 255, 255), -1)
                            elif dis > 14:
                                k = dict_kernel[14]
                                cv2.rectangle(frame, (20, 20), (60, dict_bar[14]), (255, 255, 255), -1)

                            kernel = np.ones((k, k), np.uint8)
                            edited_work_copy = cv2.erode(work_copy, kernel, iterations=1)
                            # edited = True

                        if len(work_copies) == 1:
                            work_copies.append(edited_work_copy)
                        else:
                            work_copies[1] = edited_work_copy

                        # draw
                        if gesture_defects == 1:
                            if switch == 0:
                                cv2.line(frame, (a[0] + 430, a[1] + 90), (b[0] + 430, b[1] + 90), (0, 255, 0), 4)
                                cv2.circle(frame, (a[0] + 430, a[1] + 90), 6, (0, 255, 0), -1)
                                cv2.circle(frame, (b[0] + 430, b[1] + 90), 6, (0, 255, 0), -1)

                        if gesture_defects == 2:
                            work_copies[0] = edited_work_copy
                            editing = False
                            pinch = False
                            operator_selected = False

        # Display the frames
        cv2.imshow('Webcam', frame)
        #cv2.imshow('Masked hand', mask)

        # Exit condition
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
