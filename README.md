# Camera-Calibration
## Preface
In this repository, we do camera calibration from scratch. For convenience, 
we will use some opencv libraries when it is nothing to do with camera calibration(e.g. reading images or finding 
chessboard corners). The purpose of this repository is to know camera calibration better.  

The way to find the chessboard corners is not the key part in this repository, so we utilized the function in OpenCV directly.

NOTE: We follow the algorithms and the notations written in chapter 4 in reference [2].
It doesn't optimize the code(may be slow) but keep the order the same as in reference [2].

## Support Models
Camera Number: Monocular only  
Camera(Projection) Model: pinhole  
Distortion Model: radial distortion  

## Terms In Code
In this repository, we follow the list of symbols and notation defined in table 1 in reference [2] and 
always use terms 'model points' and 'sensor points'.  
Some other common terms refer to the same things are defined as follows: 
1. Points in 3D world coordinate: model points, target points, object points
2. Points in 2D image(sensor) coordinate: image points, sensor points

## Result
| Focal Length | Principal Point | Skew | Distortion | Reprojection Error(avg.) |
| :----------: | :-------------: | :--: | :--------: | :----------------------: |
| (534.744914, 534.993645) | (342.223774, 232.569257) | 0.598208 | (-0.298821, 0.138997) | 0.143098 |

**NOTE: The result will a litte different because of the result of LM algorithm may be different.**

## Future
If someone wants to improve the result more, the following reference may help:  
1. Liu, Zhen, et al. "High-accuracy calibration of low-cost camera using image disturbance factor." Optics Express 24.21 (2016): 24321-24336.
2. Datta, Ankur, Jun-Sik Kim, and Takeo Kanade. "Accurate camera calibration using iterative refinement of control points." 2009 IEEE 12th International Conference on Computer Vision Workshops, ICCV Workshops. IEEE, 2009.

## Reference
### Theory
1. Zhang, Zhengyou. "A flexible new technique for camera calibration." IEEE Transactions on pattern analysis and machine intelligence 22.11 (2000): 1330-1334.
2. Burger, Wilhelm. "Zhang’s camera calibration algorithm: in-depth tutorial and implementation." HGB16-05 (2016): 1-6.
3. [How SVD solve the direct linear transformation(DLT) problem for estimating the homography in step 1?](https://math.stackexchange.com/questions/772039/how-does-the-svd-solve-the-least-squares-problem/2173715#2173715)
4. [Basic concepts of the homography explained with code - OpenCV](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html#lecture_16)
5. [CSE/EE486 Computer Vision I, Robert Collins](https://www.cse.psu.edu/~rtc12/CSE486/)

### Implementation
1. [Find chessboard corners](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
2. [Camera Calibration From Scratch - Python](https://github.com/goldbema/CameraCalibration)
