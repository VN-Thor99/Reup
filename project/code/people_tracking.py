import cv2
from object_detection import ObjectDetection
import math

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("Store_10s.mp4")

# Initialize count
width = int(cap.get(3))
height = int(cap.get(4))
count = 0
center_points_prev_frame = []
drink =[]
pt_drink = []
fastfood = []
pt_fastfood = []
tracking_objects = {}
track_id = 0
number_fastfood = 0
number_drink = 0
wrt = cv2.VideoWriter("out23.mp4",cv2.VideoWriter_fourcc(*'mp4v'),20,(width,height))

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    print(count)
    #Create ROI
    ROI1=cv2.rectangle(frame,(1000,200),(width,500),(0,0,255),3)
    ROI2=cv2.rectangle(frame,(520,0),(850,200),(0,0,255),3)
    cv2.putText(frame, "Drinks:", (1010,230), 0, 1, (0, 0, 255) , 4)
    cv2.putText(frame, "Fast food:", (530,30), 0, 1, (0, 0, 255), 4)
    # Point current frame
    center_points_cur_frame = []


    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        if count%10==0:
            if pt[0] > 1000 and pt[0] < width and pt[1] > 200 and pt[1] < 500:
                drink.append(object_id)
                drink = list(set(drink))
                number_drink = len(drink) - 1
            if pt[0] > 520 and pt[0] < 850 and pt[1] > 0 and pt[1] < 200:
                fastfood.append(object_id)
                fastfood = list(set(fastfood))
                number_fastfood = len(fastfood) - 1

    cv2.putText(frame, str(number_drink), (1150,230), 0, 1, (0, 0, 255) , 4)
    cv2.putText(frame, str(number_fastfood), (700,30), 0, 1, (0, 0, 255), 4)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()
    # #Save output
    # wrt.write(frame)

    key = cv2.waitKey(1)
    if key == "q":
        break

cap.release()
cv2.destroyAllWindows()
