import copy
import cv2
import numpy as np
import scipy.fftpack as fp
import os
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', default='./1tagvideo.mp4', type=str)
args = parser.parse_args()
vid_path = args.video

def plot_FFT(resized_frame):
    x = resized_frame.real.ravel()
    y = resized_frame.imag.ravel()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='white',size=1)))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,1)')
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.show()


#vid_path = './1tagvideo.mp4'
vidObj = cv2.VideoCapture(vid_path)
success = True
i=0
while success:
    success, frame = vidObj.read()
    i+=1
    if success and (i == 13):
        break

def detect_tag(frame):
    #Threshold the image, so that the details in the background won't pop up as edges
    threshold = 100
    bin_frame = copy.copy(frame)
    bin_frame[frame > threshold] = 255
    bin_frame[frame <= threshold] = 0
    if __name__ == '__main__':
        cv2.imwrite('bin_frame.jpg', bin_frame)
        cv2.imshow('bin_frame', bin_frame)
        cv2.waitKey(0)

    fft = np.fft.fft2(bin_frame)
    fft_shifted = np.fft.fftshift(fft) #shifting the fft such that zero frequency component comes to the center of the 2d array
    mask_side_len = 500 #creating a square mask of side 500
    center_x = np.shape(fft_shifted)[0]//2
    center_y = np.shape(fft_shifted)[1]//2
    fft_shifted_masked = fft_shifted
    fft_shifted_masked[int(center_x-mask_side_len/2):int(center_x+mask_side_len/2), int(center_y-mask_side_len/2):int(center_y+mask_side_len/2)] = 0

    ifft = np.fft.ifft2(np.fft.ifftshift(fft_shifted_masked)).real
    #print(ifft.shape)
    if __name__ == '__main__':
        cv2.imwrite('ifft.jpg', ifft)
        cv2.imshow('ifft', ifft)
        cv2.waitKey(0)
    sob = cv2.Sobel(ifft,cv2.CV_8U,1,1,ksize=3)
    sob = cv2.cvtColor(sob, cv2.COLOR_BGR2GRAY)
    sob[sob<100] = 0 #binary thresholding to remove fringe artifacts
    sob[sob>=100] = 255
    if __name__ == '__main__':
        cv2.imwrite('sobel.jpg', sob)
        cv2.imshow('sobel', sob)
        cv2.waitKey(0)
    return sob

if __name__ == '__main__':
    sobel_op = detect_tag(frame)