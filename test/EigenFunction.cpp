//
// Created by aqiu on 2021/1/27.
//

#include "EigenFunction.h"
#include <iostream>
#include <Eigen/Dense>

void QR()
{
    Eigen::MatrixXf A(Eigen::MatrixXf::Random(5,3));
    Eigen::MatrixXf thinQ(Eigen::MatrixXf::Identity(5,3));
    Eigen::MatrixXf Q;
    A.setRandom();
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
    Q = qr.householderQ();
    thinQ = qr.householderQ() * thinQ;
    std::cout << "The complete unitary matrix Q is:\n" << Q << "\n\n";
    std::cout << "The thin matrix Q is:\n" << thinQ << "\n\n";

    Eigen::MatrixXf R = qr.matrixQR().triangularView<Eigen::Upper>();

    std::cout << "QR is:\n" << qr.matrixQR()  << "\n\n";
    std::cout << "upper of R is:\n" << R  << "\n\n";
    std::cout << "top 3 of R is:\n" << R.topRows(3)  << "\n\n";
    // std::cout << "Q(:,1) is:\n" << R  << "\n\n";

    std::cout << " pause " << std::endl;
}

// 测试四元素积分，以下为单位四元数，旋转一周后（M_PI/10 * 20 = 2*M_PI），还为单位矩阵
void ImuInt()
{
     Eigen::Quaterniond Qwb;
     Qwb.setIdentity();
     Eigen::Vector3d omega (0,0,M_PI/10);
     double dt_tmp = 0.005;
     for (double i = 0; i < 20.; i += dt_tmp) {
         Eigen::Quaterniond dq;
         Eigen::Vector3d dtheta_half =  omega * dt_tmp /2.0;
         dq.w() = 1;
         dq.x() = dtheta_half.x();
         dq.y() = dtheta_half.y();
         dq.z() = dtheta_half.z();
         Qwb = Qwb * dq;
     }
     std::cout << Qwb.coeffs().transpose() <<"\n"<<Qwb.toRotationMatrix() << std::endl;
}
