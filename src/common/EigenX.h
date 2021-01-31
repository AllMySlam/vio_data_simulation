// EigenX.h
//
// Created by aqiu on 2021/1/31.
//

#ifndef IMUSIMWITHPOINTLINE_EIGENX_H
#define IMUSIMWITHPOINTLINE_EIGENX_H

#include <Eigen/Dense>

typedef float EigenX;

typedef Eigen::Matrix<EigenX, 2, 1> V2X;
typedef Eigen::Matrix<EigenX, 3, 1> V3X;
typedef Eigen::Matrix<EigenX, 4, 1> V4X;
typedef Eigen::Matrix<EigenX, 2, 2> M2X;
typedef Eigen::Matrix<EigenX, 3, 3> M3X;
typedef Eigen::Matrix<EigenX, 4, 4> M4X;
typedef Eigen::Matrix<EigenX, 4, 2> M42X;

typedef Eigen::VectorXf VXX;
typedef Eigen::MatrixXf MXX;

typedef Eigen::Transform<EigenX,3, Eigen::Isometry> ISO3X; //实质为4*4的矩阵
typedef Eigen::AngleAxis<EigenX> AxisX;


#endif //IMUSIMWITHPOINTLINE_EIGENX_H
