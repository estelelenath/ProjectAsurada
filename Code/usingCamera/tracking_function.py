import math
import time

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Store the bounding box size of the detected objects
        self.box_size = {}
        # Store the distance between User's Vehicle and other Vehicle
        self.distance_from_other = {}

        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

        self.space_difference = 0
        self.distance_difference_from_other_vehicle = 0            # distance change between user's vehicle and other vehicles

        self.user_vehicle_point = (540, 720)  # Case for Input Video Size(w:1080, h:720)

    def update_r(self, objects_rect):  # arg : detected object rectengles each frame
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            #x, y, w, h = rect
            #cx = (x + x + w) // 2
            #cy = (y + y + h) // 2
            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Distance between User's Vehicle and other Vehicle

            # Find out if that object was detected already
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    # calculate the changing in size of the bounding box, more details methods and comparison in GitHub
                    self.space_difference = (w * h - self.box_size[id]) / w * h
                    self.space_difference = round(self.space_difference, 2)
                    self.box_size[id] = w * h

                    self.distance_difference_from_other_vehicle = math.dist((cx, cy), self.user_vehicle_point) - self.distance_from_other[id]
                    self.distance_difference_from_other_vehicle = round(self.distance_difference_from_other_vehicle, 2)
                    self.distance_from_other[id] = math.dist((cx, cy), self.user_vehicle_point)

                    objects_bbs_ids.append([x1, y1, x2, y2, self.space_difference, self.distance_difference_from_other_vehicle, id, "id_nr"])  # appended 1/4
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.box_size[self.id_count] = w * h
                self.distance_from_other[self.id_count] = math.dist((cx, cy), self.user_vehicle_point)
                #self.distance_from_other[self.id_count] = math.dist((cx, cy), self.user_vehicle_point)
                #objects_bbs_ids.append([x, y, w, h, 0, self.id_count, "id_nr"])  # appended 2/4
                objects_bbs_ids.append([x1, y1, x2, y2, 0, 0, self.id_count, "id_nr"])  # appended 2/4
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        new_space_size = {}
        new_distance_difference = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, object_id, _ = obj_bb_id  # appended 3/4 -> main
            center = self.center_points[object_id]
            space_size = self.box_size[object_id]
            distance_difference = self.distance_from_other[object_id]

            new_center_points[object_id] = center
            new_space_size[object_id] = space_size
            new_distance_difference[object_id] = distance_difference

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.box_size = new_space_size.copy()
        self.distance_from_other = new_distance_difference.copy()
        return objects_bbs_ids

    def update_f(self, objects_rect):  # arg : detected object rectengles each frame
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            #x, y, w, h = rect
            #cx = (x + x + w) // 2
            #cy = (y + y + h) // 2
            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Distance between User's Vehicle and other Vehicle

            # Find out if that object was detected already
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    # calculate the changing in size of the bounding box, more details methods and comparison in GitHub
                    self.space_difference = (w * h - self.box_size[id]) / w * h
                    self.space_difference = round(self.space_difference, 2)
                    self.box_size[id] = w * h

                    self.distance_difference_from_other_vehicle = math.dist((cx, cy), self.user_vehicle_point) - self.distance_from_other[id]
                    self.distance_difference_from_other_vehicle = round(self.distance_difference_from_other_vehicle, 2)
                    self.distance_from_other[id] = math.dist((cx, cy), self.user_vehicle_point)

                    objects_bbs_ids.append([x1, y1, x2, y2, self.space_difference, self.distance_difference_from_other_vehicle, id, "id_nr"])  # appended 1/4
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.box_size[self.id_count] = w * h
                self.distance_from_other[self.id_count] = math.dist((cx, cy), self.user_vehicle_point)
                #self.distance_from_other[self.id_count] = math.dist((cx, cy), self.user_vehicle_point)
                #objects_bbs_ids.append([x, y, w, h, 0, self.id_count, "id_nr"])  # appended 2/4
                objects_bbs_ids.append([x1, y1, x2, y2, 0, 0, self.id_count, "id_nr"])  # appended 2/4
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        new_space_size = {}
        new_distance_difference = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, object_id, _ = obj_bb_id  # appended 3/4 -> main
            center = self.center_points[object_id]
            space_size = self.box_size[object_id]
            distance_difference = self.distance_from_other[object_id]

            new_center_points[object_id] = center
            new_space_size[object_id] = space_size
            new_distance_difference[object_id] = distance_difference

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.box_size = new_space_size.copy()
        self.distance_from_other = new_distance_difference.copy()
        return objects_bbs_ids