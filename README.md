# Multiview 3D Pose Estimation

## Semester Project @ Computer Vision Lab, EPFL.

## Check the PDF Report for more details.

## Introduction 

3D Human Pose estimation is a critical task in computer vision, as it involves determining the position and orientation of a person in an image or video. It has a wide range of applications, such as in autonomous vehicles, analytics, security, and sports. In order to obtain accurate 3D human pose estimates, data acquisition setups typically involve multiple cameras capturing a scene from different angles. This allows for the reconstruction of a 3D model of the human skeleton.

Traditional methods for 3D pose estimation from 2D poses rely on algebraic optimization, which is a computationally intensive process that requires proper camera calibration and knowledge of the camera parameters, both intrinsic and extrinsic. Thus, obtaining accurate camera parameters can be labour-intensive and time-consuming.

The goal of this work is to propose and evaluate new architectures capable of learning 3D poses from 2D poses without the prior knowledge of camera parameters. The proposed method aims to overcome the limitations of traditional methods by eliminating the need for labor-intensive camera calibration and the knowledge of camera parameters. This approach aims to make 3D pose estimation more efficient, robust and widely applicable


## Dataset

The dataset used was acquired by the computer vision lab at EPFL. It consists of 5 participants each of them performed 4 movements - Squats, Lunges, planks, and pick-ups- in various forms and repetitions. The scene was captured with 4 cameras. The sampling rate was 30 frame per second. The dataset is already processed and contains both 2D and 3D poses ground truths.  

Please Refer to https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises

<p align="center">
  <img src="https://github.com/saadelmoutaouakil/3D-Multiview-Pose-Estimation/blob/master/Figures/Figure-2.png" />
</p>

## Results 
Overall, GCNs perform consistently better than MLP. The concatenation of the 4 views of the intermediate outputs presented in our method is the best performing model. It was expected that the use of concatenation would perform better than the sum because the latter don’t enable the model to learn the contribution of each view. The baseline and the concatenation method perform very well visually (Figure 10) which indicates the model’s ability to learn both the 3D structure of a human body and the camera parameters required to situate it.


MLPs on the other hand remain an interesting choice. Even though they perform many times worse on average than GCN, the visual evaluation shows that they can still learn and represent acceptable 3D human estimations. Figure 10 shows the accuracy of this simple model on chosen representations of Squats, Lunges, planks, and pick-ups. The high error on MLPs on one side, and the good visual results on the other side indicate the MLPs’ high variance.
<p align="center">
  <img src="https://github.com/saadelmoutaouakil/3D-Multiview-Pose-Estimation/blob/master/Figures/Figure-6.png" />
</p>

