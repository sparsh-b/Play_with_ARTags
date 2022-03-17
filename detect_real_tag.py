import copy
import cv2
import numpy as np
import argparse
#from detect_tag import detect_tag
from estimate_tag import reflection_of_point, warp, angle_of_diagonal, obtain_grid, grid_decode
from math import sqrt, atan2, degrees, pi, cos, sin

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='./1tagvideo.mp4', type=str)
args = parser.parse_args()
vid_path = args.video
debug = False

def read_video(vid_path):
    vidObj = cv2.VideoCapture(vid_path)
    success = True
    i=0
    all_edges = []
    frames = []
    while success:
        success, frame = vidObj.read()
        if success:# and i>120:
            frames.append(frame)
        # if i==140:#80
        #     #frames.append(frame)
        #     # cv2.imshow('frame{}'.format(i), frame)
        #     # cv2.waitKey(0)
        #     # cv2.imwrite('frame{}.jpg'.format(i), frame)
        #     #if i==140:#81
        #         break
        i += 1
    print('read {} frames fromn video'.format(len(frames)))
    return frames

def homography(wrld_corners):
    
    cam_corners = np.array(cam_corners)
    # = np.array([[0,0], [8*cell_size-1,0], [0,8*cell_size-1], [8*cell_size-1,8*cell_size-1]])
    assert cam_corners.shape == wrld_corners.shape
    A = []
    for i in range(cam_corners.shape[0]):#x is col; y is row
        cam_x, cam_y = cam_corners[i]
        wrld_x, wrld_y = wrld_corners[i]
        A.append([-cam_x, -cam_y, -1, 0, 0, 0, wrld_x*cam_x, wrld_x*cam_y, wrld_x])
        A.append([0, 0, 0, -cam_x, -cam_y, -1, wrld_y*cam_x, wrld_y*cam_y, wrld_y])
    u, d, vt = np.linalg.svd(A)
    print(vt[-1].shape)
    h = vt[-1]
    H = h.reshape((3,3))
    H = H / H[2,2]
    return H

def euclidean_distance(point1, point2):
    assert (isinstance(point1, list) or isinstance(point1, np.ndarray)) and (isinstance(point2, list) or isinstance(point2, np.ndarray))
    return sqrt(np.sum(np.square(np.subtract(point1, point2))))

def imshow_components(labels, unique_label):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    cv2.imshow('label-{}'.format(unique_label), labeled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def order_CW(point1, point2, point3, point4, midpoint):#points are in XY coordinates (not row, column numbers)
    # returns points in coord system as well
    angle1 = atan2(point1[1]-midpoint[1], point1[0]-midpoint[0])
    angle2 = atan2(point2[1]-midpoint[1], point2[0]-midpoint[0])
    angle3 = atan2(point3[1]-midpoint[1], point3[0]-midpoint[0])
    angle4 = atan2(point4[1]-midpoint[1], point4[0]-midpoint[0])
    sorted_indices = np.argsort([angle1, angle2, angle3, angle4])[::-1]
    #print([angle1, angle2, angle3, angle4], sorted_indices)
    points_cw = np.array([point1, point2, point3, point4])[sorted_indices]
    #print(points_cw)
    return points_cw

def estimate_and_order(curr_2_points):
    # estimates the 2 points on the other diagonal as rownumber, columnnumber
    # and returns all the points arranged in Clock Wise order
    coords1 = [curr_2_points[1][1], -curr_2_points[1][0]]
    coords2 = [curr_2_points[0][1], -curr_2_points[0][0]]
    
    curr_midpoint = [(coords1[0]+coords2[0])/2, (coords1[1]+coords2[1])/2]
    half_dist = sqrt(np.sum(np.square(np.subtract(coords2, coords1))))/2
    curr_angle = atan2((coords1[1]-coords2[1]), (coords1[0]-coords2[0]))
    if curr_angle < 0:
        curr_angle += pi
    other2_angle = curr_angle + pi/2#angle that the other diagonal makes with the x axis
    new_x = curr_midpoint[0] + half_dist * cos(other2_angle)
    new_y = curr_midpoint[1] + half_dist * sin(other2_angle)    
    new_point1_coords = [new_x, new_y]
    new_point2_coords = reflection_of_point(new_point1_coords, coords1, coords2)    
    new_y *= -1 #converting back from coordinate system to row number
    new_point1_rowcol = [int(new_y), int(new_x)]
    points_cw = order_CW(coords1, coords2, new_point1_coords, new_point2_coords, curr_midpoint).astype(int)
    points_cw = [[-points_cw[0][1], points_cw[0][0]], [-points_cw[1][1], points_cw[1][0]], [-points_cw[2][1], points_cw[2][0]], [-points_cw[3][1], points_cw[3][0]]]
    return points_cw#[new_point1_rowcol, [-new_point2_coords[1], new_point2_coords[0]]]

def get_corners(frames):
    corners = []
    for frame_num in range(len(frames)):
        print('frame_num:', frame_num)
        frame = frames[frame_num]
        #apply canny
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, edges = cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY)
        low = 20
        high = 50
        edges = cv2.Canny(image=edges, threshold1=low, threshold2=high)
        cv2.imwrite('canny_edges/{}.jpg'.format(frame_num), edges)
        if debug:
            cv2.imshow('canny_edges', edges)
            cv2.waitKey(0)
        kernel2 = np.ones((3,3), np.uint8)
        img_dilation = cv2.dilate(edges, kernel2, iterations=3)
        img_erosion = cv2.erode(img_dilation, kernel2, iterations=2)
        minLineLength = 2
        maxLineGap = 100
        lines = cv2.HoughLinesP(img_erosion,1,np.pi/180,100,minLineLength,maxLineGap)
        if lines is not None:
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(img_erosion,(x1,y1),(x2,y2),(0,255,0),12)
            img_dilation = cv2.dilate(img_erosion, kernel2, iterations=5)
            img_erosion = cv2.erode(img_dilation, kernel2, iterations=2)
        
        if debug:
            cv2.imshow('edges', img_erosion)
            cv2.waitKey(0)
        

        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(img_erosion, 8, cv2.CV_32S)
        values, counts = np.unique(labels, return_counts=True)
        num_ignore_labels = 0
        consider_labels = []
        #imshow_components(labels, 'all_components')
        cluster_size_threshold_low = 300
        cluster_size_threshold_high = 80000
        for i in range(counts.shape[0]):#filter clusters based on size
            if counts[i] < cluster_size_threshold_low or counts[i] > cluster_size_threshold_high:
                num_ignore_labels += 1
                labels[labels == values[i]] = 0
            else:
                print(stats[values[i], cv2.CC_STAT_AREA])
                consider_labels.append(values[i])
        #imshow_components(labels, 'size_threshold')

        unique_labels = np.unique(labels)
        #print(num_labels, unique_labels, consider_labels)
        if debug:
            imshow_components(labels, 'centroid_threshold')

        #since all the contours are subsets/supersets of each other, we can say that the contour with the biggest area is superset of a contour of smaller area
        bbox_areas = []
        for i in consider_labels:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            bbox_areas.append(w*h)
        bbox_areas = np.array(bbox_areas)
        # The countour with the largest bbox corresponds to the edges of white paper
        # The countour with the 2nd largest bbox corresponds to the edges of the 8x8 marker
        # So to get the contour of the 8x8 marker, all we need to do is find the cpnnected component with 2nd largest bbox
        top2_indices = np.argpartition(bbox_areas, -2)[-2:]
        top2 = bbox_areas[top2_indices]
        top2_indices_sorted = top2_indices[np.argsort(top2)]
        tag_idx_in_array = top2_indices_sorted[-2]
        tag_contour_label_num = consider_labels[tag_idx_in_array]
        tag_contour = copy.copy(img_erosion)
        tag_contour[labels == tag_contour_label_num] = 255
        tag_contour[labels != tag_contour_label_num] = 0
        frame_copy = copy.copy(frame)
        cv2.imwrite('contours1/tag_contour{}.jpg'.format(frame_num), tag_contour)
        if debug:
            cv2.imshow('contours/tag_contour{}.jpg'.format(frame_num), tag_contour)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        tag_contour = np.array(tag_contour.tolist()).astype(np.int32)

        rows, cols = tag_contour.shape
        tag_contour_points = np.dstack(np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij'))[tag_contour == 255]
        tag_contour_points = np.expand_dims(tag_contour_points, 1)
        print(tag_contour_points.shape)
        for mul_factor in [0.5]:#np.linspace(10, 0.01, 1000):#[0.5]:
            hyp_param = mul_factor*cv2.arcLength(tag_contour_points, False)
            approx_contour = cv2.approxPolyDP(tag_contour_points, hyp_param, False).squeeze().tolist()
            if (len(approx_contour)) != 2:
                print('number of vertices:', len(approx_contour))
            points_cw = estimate_and_order(approx_contour)
            corners.append(points_cw)
            for crnr in range(len(points_cw)):
                cv2.circle(img_erosion, tuple(points_cw[crnr][::-1]), 15, (255,0,0))
            cv2.imwrite('contours1/img_erosion{}.jpg'.format(frame_num), img_erosion)
            if debug:
                cv2.imshow('contours/img_erosion{}.jpg'.format(frame_num), img_erosion)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return corners

def find_orientation_1st_frame(corners4, rgb_frame, frame_num):
    angle = angle_of_diagonal(corners4)#returns the angle made by the left-tilted diagonal with +X cartesian axis
    print(corners4, angle)
    # due to the order followed while forming corners4, `angle` always lies b/w +90 to +180 degrees
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    # cell_size = int(euclidean_distance(corners4[0], corners4[1])/8)
    center_of_rot = np.divide(np.add(corners4[0], corners4[2]), 2)
    angle_acw = angle-45
    M = cv2.getRotationMatrix2D(tuple(center_of_rot), angle_acw, 1.0)
    warp_tag = warp(M, gray_frame, gray_frame.shape[0], gray_frame.shape[1])
    # cv2.imwrite('q2a/warp_tag_{}.jpg'.format(frame_num), warp_tag)
    # #cv2.imshow('warp_tag', warp_tag)
    # #cv2.waitKey(0)
    warped_corners4 = []
    for crnr in range(len(corners4)):
        new_i, new_j = M.dot(np.array([corners4[crnr][0], corners4[crnr][1], 1])).astype(int)
        warped_corners4.append([new_i, new_j])
    warped_corners4 = np.array(warped_corners4)
    #print(warped_corners4)
    #warped_corners4 = order_CW(warped_corners4[0], warped_corners4[1], warped_corners4[2], warped_corners4[3], center_of_rot)
    #print(warped_corners4)
    top, left = np.amin(warped_corners4, axis=0)#[np.max([warped_corners4[0][0], warped_corners4[1][0]]), np.max([warped_corners4[0][1], warped_corners4[2][1]])]
    bottom, right = np.amax(warped_corners4, axis=0)#[np.min([warped_corners4[2][0], warped_corners4[3][0]]), np.min([warped_corners4[1][1], warped_corners4[3][1]])]
    cropped_warp = warp_tag[top:bottom, left:right]
    # cv2.imshow('cropped_warp', cropped_warp)
    # cv2.waitKey(0)
    grid = obtain_grid(cropped_warp)
    _, num_acw_90_rots = grid_decode(grid, cropped_warp)
    total_acw_rotation = (angle_acw + (90 * num_acw_90_rots))%360
    # M = cv2.getRotationMatrix2D(tuple(np.mean(warped_corners4, axis=0).astype(int)), total_acw_rotation, 1.0)
    # cropped_warp_tag = warp(M, cropped_warp, cropped_warp.shape[0], cropped_warp.shape[1])
    # cv2.imshow('q2a/warp_tag_{}.jpg'.format(frame_num), cropped_warp_tag)
    # cv2.waitKey(0)


    M = cv2.getRotationMatrix2D(tuple(center_of_rot), total_acw_rotation, 1.0)
    warp_tag_final = warp(M, gray_frame, gray_frame.shape[0], gray_frame.shape[1])
    cv2.imshow('1st_img_warp_tag', warp_tag_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return angle_acw, num_acw_90_rots


def find_orientation_rest_frames(corners4, rgb_frame, frame_num, prev_acw_angle, prev_acw_90_rots):
    angle = angle_of_diagonal(corners4)
    # print(corners4, angle)
    # return
    center_of_rot = np.divide(np.add(corners4[0], corners4[2]), 2)
    angle_acw = angle-45
    angle_threshold = 10#max angle the marker rotates between successive frames. 
    if angle_acw <= prev_acw_angle: #marker turned CW or (turned ACW & shifted to next quadrant)
        if abs(prev_acw_angle - angle_acw) <= angle_threshold:
            M = cv2.getRotationMatrix2D(tuple(center_of_rot), (angle_acw + 90* prev_acw_90_rots)%360, 1.0)
        else:
            prev_acw_90_rots += 1
            M = cv2.getRotationMatrix2D(tuple(center_of_rot), (angle_acw + 90* prev_acw_90_rots)%360, 1.0)
        warp_tag = warp(M, rgb_frame, rgb_frame.shape[0], rgb_frame.shape[1], rgb_frame.shape[2])
        cv2.imwrite('q2a/final_warp_tag_{}.jpg'.format(frame_num), warp_tag)

        # cv2.imshow('warp_tag', warp_tag)
        # cv2.waitKey(0)
    else: #marker turned ACW or (turned CW & shifted to next quadrant)
        if abs(prev_acw_angle - angle_acw) <= angle_threshold:
            M = cv2.getRotationMatrix2D(tuple(center_of_rot), (angle_acw + 90* prev_acw_90_rots)%360, 1.0)
        else:
            prev_acw_90_rots -= 1
            M = cv2.getRotationMatrix2D(tuple(center_of_rot), (angle_acw + 90* prev_acw_90_rots)%360, 1.0)
        warp_tag = warp(M, rgb_frame, rgb_frame.shape[0], rgb_frame.shape[1], rgb_frame.shape[2])
        cv2.imwrite('q2a/final_warp_tag_{}.jpg'.format(frame_num), warp_tag)

        # cv2.imshow('warp_tag', warp_tag)
        # cv2.waitKey(0)
    print(prev_acw_angle, angle_acw, prev_acw_90_rots, frame_num)
    return angle_acw, prev_acw_90_rots



if __name__ == '__main__':
    frames = read_video(vid_path)
    corners = get_corners(frames)#corners are sorted in CW order & first corner in each of the 1D lists is the one in Quadrant2 (when midpoint is the origin & X,Y axes directions are in the 2D-cartesian sense)
    #print(corners)
    testudo = cv2.imread('testudo.png')
    next_acw_angle, next_acw_90_rots = find_orientation_1st_frame(corners[0], frames[0], 0)
    total_acw_angle = next_acw_angle + 90 * next_acw_90_rots
    
    for i in range(1, len(frames)):
        prev_acw_angle = next_acw_angle
        prev_acw_90_rots = next_acw_90_rots
        next_acw_angle, next_acw_90_rots = find_orientation_rest_frames(corners[i], frames[i], i, prev_acw_angle, prev_acw_90_rots)