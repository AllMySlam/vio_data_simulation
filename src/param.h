//
// Created by hyj on 17-6-22.
//

#ifndef IMUSIM_PARAM_H
#define IMUSIM_PARAM_H

#include <eigen3/Eigen/Core>
#define FENCE 0

class Param{

public:

    Param();

    // time
    int imu_frequency = 200;
    int cam_frequency = 30;
    double imu_timestep = 1./imu_frequency;
    double cam_timestep = 1./cam_frequency;
    double t_start = 0.;
    double t_end = 20;  //  20 s

    // noise
    double gyro_bias_sigma = 1.0e-5; // gyroscope bias random work noise standard deviation.     #4.0e-5
    double acc_bias_sigma = 0.0001; // accelerometer bias random work noise standard deviation.  #0.02

    double gyro_noise_sigma = 0.015;    // rad/s * 1/sqrt(hz) # accelerometer measurement noise standard deviation. #0.2   0.04
    double acc_noise_sigma = 0.019;      //　m/(s^2) * 1/sqrt(hz) # gyroscope measurement noise standard deviation.     #0.05  0.004

    double pixel_noise = 1;              // 1 pixel noise

    // cam f
    double fx = 460;
    double fy = 460;
    double cx = 255;
    double cy = 255;
    double image_w = 640;
    double image_h = 640;


    // 外参数
    Eigen::Matrix3d R_bc;   // cam to body
    Eigen::Vector3d t_bc;     // cam to body

};


#endif //IMUSIM_PARAM_H
