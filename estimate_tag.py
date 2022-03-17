import copy
import math
import cv2
from math import sqrt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./rotated_image.jpg', type=str)
args = parser.parse_args()


def obtain_grid(tag):
    if __name__ != '__main__':
        cell_size = int(tag.shape[0]/8)
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

def grid_decode(grid, tag):
    num_rotations = 0
    while grid[5, 5] != 1:#rotate the matrix until bottom-right is white
        grid = np.rot90(grid)
        tag = np.rot90(tag)
        num_rotations += 1
    grid = grid.astype(np.uint8)
    id_binary = '0b'+str(grid[3,3])+str(grid[3,4])+str(grid[4,4])+str(grid[4,3])
    if __name__ == '__main__':
        cv2.imwrite('upright_warp_tag.jpg', tag)
    # cv2.imshow('upright_warp_tag', tag)
    # cv2.waitKey(0)
    return int(id_binary, 2), num_rotations

def euclidean_distance(point1, point2):
    assert (isinstance(point1, list) or isinstance(point1, np.ndarray)) and (isinstance(point2, list) or isinstance(point2, np.ndarray))
    return sqrt(np.sum(np.square(np.subtract(point1, point2))))

def reflection_of_point(point, line_point1, line_point2):#takes point, line_point1 & 2 represented in coordinate system (0th idx is column number of the pixel & 1st index is -row number)
    #returns the reflectionin coordinate system. Have to convert it into row & column number b4 using
    a = line_point1[1] - line_point2[1]
    b = line_point2[0] - line_point1[0]
    c = -(a*line_point1[0] + b*line_point1[1])
    reflection_point = (np.array([[b**2 - a**2, -2 * a * b], [-2 * a * b, a**2 - b**2]]) @ point - 2 * c * np.array([a, b])) / (a**2 + b**2)
    reflection_point = reflection_point.astype(int)
    return reflection_point

def angle_of_diagonal(corners4):
    distances = [-1]
    for i in range(1, 4): #selecting the line joining any 2 opposite corners & determining the agnle to be rotatted to make its angle=npi/4
        distances.append(euclidean_distance(corners4[0], corners4[i]))
    point1, point2 = corners4[0], corners4[np.argmax(distances)]
    # coords1 = [point1[0], -point1[1]]
    # coords2 = [point2[0], -point2[1]]
    coords1 = [point1[1], -point1[0]]
    coords2 = [point2[1], -point2[0]]
    angle = math.degrees(math.atan2((coords1[1]-coords2[1]), (coords1[0]-coords2[0])))
    return angle

def detect_corners(tag, rgb_tag):
    corners = cv2.goodFeaturesToTrack(tag,20,0.01,10)
    corners = np.int0(corners)
    corners = np.squeeze(corners)
    corners10 = []
    center = [cell_size*4, cell_size*4]
    delta = .20*cell_size
    print('cell_size, delta', cell_size, delta)
    for corner in corners:
        if euclidean_distance(corner, center) > sqrt(2)*cell_size-delta and euclidean_distance(corner, center) < 2*sqrt(2)*cell_size+delta:
            corners10.append(corner.tolist())
    corners10_bckp = corners10.copy()
    corners4 = []
    while len(corners10) != 0:
        ref_point = corners10[0]
        group = [ref_point]
        corners10.pop(0)

        for i in range(len(corners10)):
            if euclidean_distance(corners10[i], ref_point) <= sqrt(2)*cell_size+delta:
                group.append(corners10[i])
                flag = 1
        
        if len(group) > 1:
            for point in group[1:]:
                corners10.pop(corners10.index(point))
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
            corners4.append([-vertex4[1], vertex4[0]])
        else:
            corners4.append(group[0])
    
    angle = angle_of_diagonal(corners4)
    angle_2_rotate = 45-angle

    tag_corners =copy.copy(rgb_tag)
    for i in corners:
        x,y = i
        cv2.circle(tag_corners,(x,y),5,(0,0,255),-1)
    cv2.imwrite('corners_all.jpg', tag_corners)

    tag_corners10 =copy.copy(rgb_tag)
    for i in corners10_bckp:
        x,y = i
        cv2.circle(tag_corners10,(x,y),5,(0,0,255),-1)
    cv2.imwrite('corners10.jpg', tag_corners10)

    print('4 corners of the tag:')
    for i in corners4:
        print(i)
        x,y = i
        cv2.circle(rgb_tag,(x,y),5,(0,0,255),-1)
    cv2.imwrite('corners4.jpg', rgb_tag)
    return corners4, angle_2_rotate

def warp(M, tag, num_rows, num_cols, num_channels=1):
    if num_channels == 1:
        warp_tag = np.zeros((num_rows, num_cols),dtype = np.uint8)
    else:
        warp_tag = np.zeros((num_rows, num_cols, num_channels),dtype = np.uint8)
    for i in range(num_rows):
        for j in range(num_cols):
            new_i, new_j = M.dot(np.array([i,j,1])).astype(int)
            #print(i, j, new_i, new_j)
            if (new_i >=0 and new_i<num_rows) and (new_j>=0 and new_j<num_cols):
                warp_tag[new_i, new_j] = tag[i, j]
    return warp_tag

def decode_tag(tag, rgb_tag, cellsize):
    global cell_size
    cell_size = cellsize
    corners, angle_2_rotate = detect_corners(tag, rgb_tag) #detect the 4 corners of the tag
    print('angle_2_rotate', angle_2_rotate)
    M = cv2.getRotationMatrix2D((cell_size*8/2, cell_size*8/2), angle_2_rotate, 1.0)
    warp_tag = warp(M, tag, cell_size*8, cell_size*8)
    # cv2.imshow('warp_tag', warp_tag)
    # cv2.waitKey(0)
    if __name__ == '__main__':
        cv2.imwrite('warp_tag.jpg', warp_tag)
    grid = obtain_grid(warp_tag)
    id, num_90_rotations = grid_decode(grid, warp_tag)
    print('id of the tag:', id)
    if __name__ == '__main__':
        print('Saved different stages of the solution!')
    return angle_2_rotate + num_90_rotations*90

if __name__ == '__main__':
    rgb_tag = cv2.imread(args.image)
    tag = cv2.imread(args.image, 0)
    cell_size = int(624/8) #assuming the image size is 624 based on the image given in pdf
    if cell_size != tag.shape[0]/8 or cell_size != tag.shape[1]/8:
        tag = cv2.resize(tag, (8*cell_size, 8*cell_size),  interpolation = cv2.INTER_AREA)
        rgb_tag = cv2.resize(rgb_tag, (8*cell_size, 8*cell_size),  interpolation = cv2.INTER_AREA)
    _ = decode_tag(tag, rgb_tag, cell_size)


# H = homography(corners)
# dest = np.zeros((cell_size*8, cell_size*8, 3),dtype = np.uint8)
# warped_tag = Warp(rgb_tag, H, dest)
# print(warped_tag.shape)
# cv2.imshow('warped_tag', warped_tag)
# cv2.waitKey(0)