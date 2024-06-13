import cv2
import numpy as np


def rotz(t):
    """Rotation about the z-axis.

    :param t: rotation angle
    :return: rotation matrix
    """

    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s],
                     [s,  c]])


def get_corners_2d(box):
    """Takes an bounding box and calculate the 2D corners in BEV plane.

    0 --- 1
    |     |        x
    |     |        ^
    |     |        |
    3 --- 2  y <---o

    :param box: 3D bounding box, [x, y, z, l, w, h, r]
    :return: corners_2d: (4,2) array in left image coord.
    """

    # compute rotational matrix around yaw axis
    rz = box[6]
    R = rotz(rz)

    # 2d bounding box dimensions
    l = box[3]
    w = box[4]

    # 2d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_2d = np.dot(R, np.vstack([x_corners, y_corners]))
    corners_2d[0, :] = corners_2d[0, :] + box[0]
    corners_2d[1, :] = corners_2d[1, :] + box[1]

    return corners_2d.T


def rot_line_90(point1, point2):
    """Rotate a line around its center for 90 degree in BEV plane.

    :param point1: End point1, [x, y]
    :param point2: End point2, [x, y]
    :return: rot_line
    """

    center_x = (point1[0] + point2[0]) / 2
    center_y = (point1[1] + point2[1]) / 2
    rot_point1 = np.dot(rotz(np.pi / 2), [point1[0] - center_x, point1[1] - center_y])
    rot_point2 = np.dot(rotz(np.pi / 2), [point2[0] - center_x, point2[1] - center_y])
    rot_point1 += [center_x, center_y]
    rot_point2 += [center_x, center_y]

    return np.array([rot_point1, rot_point2])

def draw_radar_on_canvas(canvas, radar_points, bev_range, resolution=0.1, color=(0, 255, 0)):
    if len(radar_points) == 0:
        return
    
    box_centers = np.copy(radar_points[:, 0:2])
    box_centers[:, 0] -= bev_range[0]
    box_centers[:, 1] -= bev_range[1]
    box_centers /= resolution
    height = canvas.shape[0]
    width = canvas.shape[1]
    box_centers[:, 0] = height - box_centers[:, 0] 
    box_centers[:, 1] = width - box_centers[:, 1] 
    
    for idx in range(len(box_centers)):
        cv2.circle(canvas, center=(int(box_centers[idx][1]),
                int(box_centers[idx][0])), radius=4, color=color, thickness=-1)
    
def draw_boxes_on_canvas(canvas, boxes, bev_range, scores=None, label_strings=None, resolution=0.1, colors=(0, 255, 0)):
    if len(boxes) == 0:
        return
    box_centers = np.copy(boxes[:, 0:2])
    box_centers[:, 0] -= bev_range[0]
    box_centers[:, 1] -= bev_range[1]
    box_centers /= resolution
    height = canvas.shape[0]
    width = canvas.shape[1]
    box_centers[:, 0] = height - box_centers[:, 0] 
    box_centers[:, 1] = width - box_centers[:, 1] 
    for idx, box in enumerate(boxes):
        # if scores is not None and scores[idx] < 0.1:
        #     continue
        box2d = get_corners_2d(box)
        box2d[:, 0] -= bev_range[0]
        box2d[:, 1] -= bev_range[1]
        # in canvas coordinates
        p1 = (width - int(box2d[0, 1] / resolution), height - int(box2d[0, 0] / resolution))
        p2 = (width - int(box2d[1, 1] / resolution), height - int(box2d[1, 0] / resolution))
        p3 = (width - int(box2d[2, 1] / resolution), height - int(box2d[2, 0] / resolution))
        p4 = (width - int(box2d[3, 1] / resolution), height - int(box2d[3, 0] / resolution))
        # Plot box
        if isinstance(colors, list):
            color = colors[idx]
        else:
            color = colors
        cv2.line(canvas, p1, p2, color, 2)
        cv2.line(canvas, p2, p3 , color, 2)
        cv2.line(canvas, p3, p4, color, 2)
        cv2.line(canvas, p4, p1, color, 2)
        # Plot heading
        heading_points = rot_line_90(p1, p2) # bit of a hack: draw heading as just the front edge rotated by 90 degrees
        # opency internal type stuff
        heading_points = ((int(heading_points[0][0]), int(heading_points[0][1])), (int(heading_points[1][0]), int(heading_points[1][1])))
        cv2.line(canvas, heading_points[0],
                 heading_points[1], color, thickness=2)

        text = ""
        if label_strings is not None:
            text += label_strings[idx]
        if scores is not None:
            if label_strings is not None:
                text += ", "
            text += "%.2f" % scores[idx]
        if text:
            # note y and x are switched, due to difference in convention between imu and images
            origin = (int(box_centers[idx][1]), int(box_centers[idx][0]))
            cv2.putText(canvas, text, origin, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, color=color, thickness=2)


det_colors_table = {
    'Car': [0, 0, 255],
    'Truck': [255, 0, 0],
    'Pedestrian': [255, 0, 255],
    'Cyclist': [255, 255, 0]
}

def plot_gt_det_cmp(points, gt_boxes, det_boxes, bev_range, scores=None, path=None, true_where_point_on_img=None, names=None, radar_points=None):
    """Visualize all gt boxes and det boxes for comparison.

    :param points: lidar points, [N, 3]
    :param gt_boxes: gt boxes, [N, [x, y, z, l, w, h, r]]
    :param det_boxes: det boxes, [N, [x, y, z, l, w, h, r]]
    :param bev_range: bev range, [x_min, y_min, z_min, x_max, y_max, z_max]
    :return:
    """

    # Configure the resolution
    resolution = 0.1
    # Initialize the plotting canvas
    pixels_x = int((bev_range[3] - bev_range[0]) / resolution) + 2
    pixels_y = int((bev_range[4] - bev_range[1]) / resolution) + 2
    canvas = np.zeros((pixels_x, pixels_y, 3), np.uint8)
    canvas.fill(0)

    # Plot the point cloud
    loc_x = ((points[:, 0] - bev_range[0]) / resolution).astype(int)
    loc_y = ((points[:, 1] - bev_range[1]) / resolution).astype(int)
    
    infov = (loc_x >=0) & (loc_x < pixels_x-2) & (loc_y < pixels_y-2) & (loc_y >=0)
    loc_x = loc_x[infov]
    loc_y = loc_y[infov]
    canvas[loc_x, loc_y] = [0, 255, 255]
    
    if true_where_point_on_img is not None:
        on_img_x = ((points[true_where_point_on_img, 0] - bev_range[0]) / resolution).astype(int)
        on_img_y = ((points[true_where_point_on_img, 1] - bev_range[1]) / resolution).astype(int)
        infov = (on_img_x >=0) & (on_img_x < pixels_x-2) & (on_img_y < pixels_y-2) & (on_img_y >=0)
        on_img_x = on_img_x[infov]
        on_img_y = on_img_y[infov]
        canvas[on_img_x, on_img_y] = [255, 255, 0]
    
    x_range = np.array([10, 25, 50, 75, 100, 125])
    y1_range = np.ones(len(x_range)) * bev_range[1]
    y2_range = np.ones(len(x_range)) * bev_range[4]
    x_range = ((x_range - bev_range[0]) / resolution).astype(int)
    y1_range = ((y1_range - bev_range[1]) / resolution).astype(int)
    y2_range = ((y2_range - bev_range[1]) / resolution).astype(int)
    for x, y1, y2 in zip(x_range, y1_range, y2_range):
        cv2.line(canvas, (y1, x), (y2, x), (0, 255, 0), 1)

    # Rotate the canvas to correct direction
    canvas = cv2.flip(canvas, 0)
    canvas = cv2.flip(canvas, 1)

    # Plot the gt boxes
    gt_color = (0, 255, 0)
    det_color = (0, 0, 255) # BGR
    
    det_colors = [det_colors_table[x] for x in names] if names is not None else (0, 255, 0)
    draw_boxes_on_canvas(canvas, gt_boxes, bev_range, resolution=resolution, colors=gt_color)
    draw_boxes_on_canvas(canvas, det_boxes, bev_range, scores=scores, resolution=resolution, colors=det_colors)
    
    if radar_points is not None:
        draw_radar_on_canvas(canvas, radar_points, bev_range, resolution, color=(0, 0, 255))
    
    color_idx = 1
    for k,v in det_colors_table.items():
        cv2.putText(canvas, k, (10, 20*color_idx), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=v, thickness=2)
        color_idx += 1

    if path is not None:
        cv2.imwrite(path, canvas)
    return canvas
