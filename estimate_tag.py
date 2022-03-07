import cv2
from math import sqrt
import numpy as np

cell_size = int(624/8) #assuming the image size is 624

def apply_homography():
    print("Didn't implement homography yet.")

def obtain_grid(tag):
    if cell_size != tag.shape[0]/8 or cell_size != tag.shape[1]/8:
        tag = cv2.resize(tag, (8*cell_size, 8*cell_size),  interpolation = cv2.INTER_AREA)
    tag = tag.astype(np.uint8)
    cells = np.ones([8, 8, cell_size, cell_size]).astype(np.uint8)
    grid = np.ones([8,8])
    for i in range(8):
        for j in range(8):
            cells[i, j] = tag[cell_size*i:cell_size*(i+1), cell_size*j:cell_size*(j+1)]
            grid[i, j] = np.argmax(np.bincount(cells[i,j].flatten()))
    grid[grid<=127] = 0
    grid[grid>127] = 1
    return grid

def grid_decode(grid):
    while grid[5, 5] != 1:#rotate the matrix until bottom-right is white
        grid = np.rot90(grid)
    grid = grid.astype(np.uint8)
    id_binary = '0b'+str(grid[3,3])+str(grid[3,4])+str(grid[4,4])+str(grid[4,3])
    return int(id_binary, 2)

def euclidean_distance(point1, point2):
    assert (isinstance(point1, list) or isinstance(point1, np.ndarray)) and (isinstance(point2, list) or isinstance(point2, np.ndarray))
    return sqrt(np.sum(np.square(np.subtract(point1, point2))))

def reflection_of_point(point, line_point1, line_point2):
    a = line_point1[1] - line_point2[1]
    b = line_point2[0] - line_point1[0]
    c = -(a*line_point1[0] + b*line_point1[1])
    reflection_point = (np.array([[b**2 - a**2, -2 * a * b], [-2 * a * b, a**2 - b**2]]) @ point - 2 * c * np.array([a, b])) / (a**2 + b**2)
    reflection_point = reflection_point.astype(int)
    return reflection_point

def detect_corners(tag, rgb_tag):
    corners = cv2.goodFeaturesToTrack(tag,20,0.01,10)
    corners = np.int0(corners)
    corners = np.squeeze(corners)
    corners10 = []
    assert tag.shape[0] == cell_size*8 and tag.shape[1] == cell_size*8
    center = [cell_size*4, cell_size*4]
    delta = 20
    for corner in corners:
        if euclidean_distance(corner, center) > sqrt(2)*cell_size-delta and euclidean_distance(corner, center) < 2*sqrt(2)*cell_size+delta:
            corners10.append(corner.tolist())
    #print(len(corners))
    #print(len(corners10))
    corners4 = []
    while len(corners10) != 0:
        ref_point = corners10[0]
        group = [ref_point]
        corners10.pop(0)
        indx_to_remove = []

        for i in range(len(corners10)):
            if euclidean_distance(corners10[i], ref_point) <= sqrt(2)*cell_size+delta:
                #print(euclidean_distance(corners10[i], ref_point), sqrt(2)*cell_size+delta)
                group.append(corners10[i])
                flag = 1
                #break
                #indx_to_remove.append(i)
        
        if len(group) > 1:
            for point in group[1:]:
                #print(point, corners10)
                corners10.pop(corners10.index(point))
        #print(ref_point, corners10)
        if len(group) == 3:
            for i in range(len(group)):
                if abs(euclidean_distance(group[i], group[i-1]) - euclidean_distance(group[i], group[(i+1)%len(group)])) <= delta/2:
                    diag_el1 = group[i-1]
                    diag_el2 = group[(i+1)%len(group)]
                    non_diag_el = group[i]
                    break
            diag_coord1 = [diag_el1[1], -diag_el1[0]]
            diag_coord2 = [diag_el2[1], -diag_el2[0]]
            non_diag_coord = [non_diag_el[1], -non_diag_el[0]]
            vertex4 = reflection_of_point(non_diag_coord, diag_coord1, diag_coord2)
            print(non_diag_coord, diag_coord1, diag_coord2, vertex4)
            corners4.append([-vertex4[1], vertex4[0]])
        else:
            corners4.append(group[0])
        
    
    for i in corners4:
        print(i)
        x,y = i
        cv2.circle(rgb_tag,(x,y),13,255,-1)
    cv2.imshow('corners', rgb_tag)
    cv2.waitKey(0)
    return corners4


rgb_tag = cv2.imread('./rotated_image.jpg')
tag = cv2.imread('./rotated_image.jpg', 0)

corners = detect_corners(tag, rgb_tag) #detect corners in the image


#apply_homography()
#grid = obtain_grid(tag)
#id = grid_decode(grid)
#print('id of the tag:', id)
exit()