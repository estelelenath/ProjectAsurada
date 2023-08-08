import math
import time


class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Store the bounding box size of the detected objects
        self.box_size = {}

        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        self.space_difference = 0



    def update(self, objects_rect):     # input : detected object rectengles from one frame?
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False


            for id, pt in self.center_points.items():
            #for id, (pt, size) in zip(self.center_points.items(), self.box_size.items()):
                #print("cx",cx, "cy",cy, "pt", pt, "pt[0]", pt[0], "pt[1]", pt[1])
                dist = math.hypot(cx - pt[0], cy - pt[1])
                #self.space_difference = w*h[0]
                #print(self.space_difference)
                #space_difference = math.hypot(w*h - size)


                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    #print("self.box_size", self.box_size)
                    #print("self.box_size[id]", self.box_size[id])
                    #print("id", id)
                    #print("w", w, "h", h, "w*h", w*h)
                    # ----------------------------------------------
                    # dx = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                    # dt = time.time() - prev_time
                    # speed = dx / dt
                    # ----------------------------------------------
                    self.space_difference = (w*h - self.box_size[id]) / w*h #calculate the changing in size of the bounding box, maybe another method is : w*h/self.box_size[id]
                    #print("space_difference", self.space_difference)
                    self.box_size[id] = w*h
                    #print("self.center_points",self.center_points)
                    #print("self.box_size",self.box_size)
                    objects_bbs_ids.append([x, y, w, h, self.space_difference, id, "id_nr"])         #appended 1/4
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                self.box_size[self.id_count] = w*h
                #print("self.center_points",self.center_points)
                #print("self.box_size",self.box_size)
                objects_bbs_ids.append([x, y, w, h, 0, self.id_count, "id_nr"])                      #appended 2/4
                ##objects_bbs_ids.append([x, y, w, h, space_difference, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        new_space_size = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id, _ = obj_bb_id                                            #appended 3/4 -> main
            center = self.center_points[object_id]
            space_size = self.box_size[object_id]

            new_center_points[object_id] = center
            new_space_size[object_id] = space_size

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        self.box_size = new_space_size.copy()
        return objects_bbs_ids