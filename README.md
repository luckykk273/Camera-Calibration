# Camera-Calibration
## Preface
In this repository, we do camera calibration from scratch. For convenience, 
we will use some opencv libraries when it is nothing to do with camera calibration(e.g. reading images or finding 
chessboard corners). The purpose of this repository is to know camera calibration better.  

The way to find the chessboard corners is not the key part in this repository, so we utilized the function in OpenCV directly.
## Support Models
### Camera Number
**NOTE: No matter how many cameras are used, you can see it as multi-monocular.**
 - Monocular only
### Camera(Projection) Model
- pinhole
### Distortion Model
- radial-tangential (radtan)
## Reference
### Theory
1. Zhang, Zhengyou. "A flexible new technique for camera calibration." IEEE Transactions on pattern analysis and machine intelligence 22.11 (2000): 1330-1334.
2. Burger, Wilhelm. "Zhang’s camera calibration algorithm: in-depth tutorial and implementation." HGB16-05 (2016): 1-6.
3. https://kushalvyas.github.io/calib.html
4. [How SVD solve the direct linear transformation(DLT) problem for estimating the homography in step 1?](https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem/2173715#2173715)
5. [Basic concepts of the homography explained with code - OpenCV](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html#lecture_16)
6. [CSE/EE486 Computer Vision I, Robert Collins](https://www.cse.psu.edu/~rtc12/CSE486/)

### Implementation
1. [Find chessboard corners](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
2. 