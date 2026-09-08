import cv2
import json


zone_points = []
zone_selected = False


def select_zone(event, x, y, flags, param):

    global zone_points, zone_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        zone_points.append((x, y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(zone_points) >= 3:
            zone_selected = True
        else:
            print("Please select at least 3 points to form a zone.")


def save_zone(zone_points):

    data = {
        "zone": zone_points
    }

    with open("zone.json", "w") as file:
        json.dump(data, file, indent=4)