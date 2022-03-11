import copy
import cv2
import numpy as np
import argparse
#from detect_tag import detect_tag
#from estimate_tag import reflection_of_point
from math import sqrt, atan2, degrees, pi, cos, sin

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='./1tagvideo.mp4', type=str)
args = parser.parse_args()
vid_path = args.video

def read_video(vid_path):
    vidObj = cv2.VideoCapture(vid_path)
    success = True
    i=0
    all_edges = []
    frames = []
    while success:
        success, frame = vidObj.read()
        # if success:# and i>120:
        #     pass        
        #     #all_edges.append(detect_tag(frame))
            # frames.append(frame)
        if i==2:
            #all_edges.append(detect_tag(frame))
            frames.append(frame)
            break
        i += 1
    print('read {} frames fromn video'.format(len(frames)))
    return frames

debug = False

def euclidean_distance(point1, point2):
    assert (isinstance(point1, list) or isinstance(point1, np.ndarray)) and (isinstance(point2, list) or isinstance(point2, np.ndarray))
    return sqrt(np.sum(np.square(np.subtract(point1, point2))))

def imshow_components(labels, unique_label):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('label-{}'.format(unique_label), labeled_img)
    #cv2.imwrite('label-{}.jpg'.format(unique_label), labeled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reflection_of_point(point, line_point1, line_point2):#takes point, line_point1 & 2 represented in coordinate system (0th idx is column number of the pixel & 1st index is -row number)
    #returns the reflectionin coordinate system. Have to convert it into row & column number b4 using
    a = line_point1[1] - line_point2[1]
    b = line_point2[0] - line_point1[0]
    c = -(a*line_point1[0] + b*line_point1[1])
    reflection_point = (np.array([[b**2 - a**2, -2 * a * b], [-2 * a * b, a**2 - b**2]]) @ point - 2 * c * np.array([a, b])) / (a**2 + b**2)
    reflection_point = reflection_point.astype(int)
    return reflection_point

def estimate_other_2_points(curr_2_points):
    #returns the estimated point as rownumber, columnnumber
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
    new_point2 = reflection_of_point([new_x, new_y], coords1, coords2)
    new_y *= -1 #converting back from coordinate system to row number
    new_point1 = [int(new_y), int(new_x)]
    return [new_point1, [-new_point2[1], new_point2[0]]]

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
        
        print('Number of lines:', len(lines))
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
        print(num_labels, unique_labels, consider_labels)
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
            other2points = estimate_other_2_points(approx_contour)
            approx_contour.append(other2points[0])
            approx_contour.append(other2points[1])
            corners.append(approx_contour)
            for crnr in range(len(approx_contour)):
                cv2.circle(img_erosion, tuple(approx_contour[crnr][::-1]), 15, (255,0,0))
            cv2.imwrite('contours1/img_erosion{}.jpg'.format(frame_num), img_erosion)
            if debug:
                cv2.imshow('contours/img_erosion{}.jpg'.format(frame_num), img_erosion)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return corners

if __name__ == '__main__':
    frames = read_video(vid_path)
    corners = get_corners(frames)
    