# Description
- This repo is done as part of the course ENPM 673 at University of Maryland, College Park.
- It detects ARTags, estimates the tag's orientation & decodes the tag.

# Libraries used:
- plotly
- numpy
- opencv
- scipy

## Instructions to Run:
### Detect the outlines of a real tag captured from camera
- Used Fourier Transform to remove the low frequency components.
- Then sobel operator is applied to detect the edges
- `python3 detect_tag.py --video='/path/to/video'`

### Estimate the tag's orientation & decode the information embedded in the tag
- First all the corners are detected.
- Then 10 corners corresponding to the edges of the tag are filtered.
- Then the 4 corners corresponding to the 4x4 tag are estimated using the 10 corners above.
- Then the orientation of the tag is estimated & its rotated into upright position so that it can be decoded.
- `python3 estimate_tag.py --image='/path/to/reference/tag'`

### Estimate the orientation & decode the information embedded in a real tag
- `python3 detect_real_tag.py --video='/path/to/video'`