//
// Created by aqiu on 2021/1/27.
//

#include "EigenX.h"
#include "EigenFunction.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
using namespace Eigen;
using namespace std;
using namespace cv;

/**
| Module      | HeaderFile                 | contents                                           |
| ----------- | -------------------------- | -------------------------------------------------- |
| Core        | #include<Eigen/Core>       | Matrix和Array的类，基础的线性代数和数组运算             |
| Geometry    | include<Eigen/Geometry>    | 旋转、平移、缩放、2维、3维                             |
| LU          | include<Eigen/LU>          | 求逆、行列式、LU分解                                  |
| Cholesky    | include<Eigen/Cholesky>    | LLT、LDLT Cholesky分解                              |
| Householder | include<Eigen/Householder> | Householder变换，用于线形代数运算                      |
| SVD         | include<Eigen//SVD>        | SVD分解                                             |
| QR          | include<Eigen/QR>          | QR分解                                              |
| Eigenvalues | include<Eigen/Eigenvalues> | 特征值、特征向量                                      |
| Sqarse      | include<Eigen/Sqarse>      | 稀疏矩阵的存储和一些基本的线性运算                       |
| Dense       | include<Eigen/Dense>       | 包含了Core/Geometry/LU/Cholesky/SVD/QR/Eigenvalues  |
| Matrix      | include<Eigen/Eigen>       | 包含了Spase和Dense（整合库）                          |
| 很多扩展库    | unsupported/Eigen          | 包括自动求导、FFT、非线性优化                          |
*/


/**
 * 测试四元素积分，以下为单位四元数，旋转一周后（M_PI/10 * 20 = 2*M_PI），还为单位矩阵
 */
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

/**
 * 1 Eigen 矩阵运算
 * http://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html
 */

/**
 * Map构造默认是按照列优先的，
 * 即按照列依次从上倒下、从左到右填充，
 * 若想换成行优先只需要在模板参数里加上RowMajor即可。
 */
void EigenMap()
{
    EigenX data[8];
    for (int i = 0; i < 8; ++i) {
        data[i] = i;
    }
    Eigen::Map<MXX> md1(data, 2, 4);
    Eigen::Map<M2X> md2(data);
    Eigen::Map<M42X> md3(data);
    Eigen::Map<Eigen::Matrix<float, 4, 2, Eigen::RowMajor>> md4(data);
    cout << md1 << endl << endl;
    cout << md2 << endl << endl;
    cout << md3 << endl << endl;
    cout << md4 << endl;
}

void EigenOther()
{
    // 1 reshape :
    // 不改变矩阵元素个数的情况下，改变矩阵中元素的大小，简单例子如转置，2x3->3x2
    MXX M1(3, 3);    // Column-major storage
    M1 << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    Eigen::Map<Eigen::RowVectorXf> v1(M1.data(), M1.size());
    cout << "v1:" << endl << v1 << endl;


    // 2 slicing : 所谓切片就是抽取矩阵中指定的一组行、列或一些元素
    Eigen::RowVectorXf v = Eigen::RowVectorXf::LinSpaced(20, 0, 19);
    cout << "Input:" << endl << v << endl;
    Eigen::Map<Eigen::RowVectorXf, 0, Eigen::InnerStride<2> > v2(v.data(), v.size() / 2);
    cout << "Even:" << v2 << endl;
}

void EigenBroadcast()
{
    // 1 Random
    int i, j;
    float maxVal;
    MXX m1 = MXX::Random(4,7);
    maxVal = m1.maxCoeff(&i, &j);
    cout << "m1:\n" << m1 << endl;
    cout << "maxVal:" << maxVal << " m("<<i<<"," << j<<") " << m1(i, j) << endl;

    // 2 Constant
    M3X m2 = M3X::Identity() + M3X::Constant(8);
    cout << "m2:\n" << m2 << endl;
    float maxIndex;
    float maxNorm = m2.colwise().sum().maxCoeff(&maxIndex);
    cout << "maxIndex:" << maxIndex << " maxNorm "<< maxNorm << endl;

    // 3 colwise
    V3X v3; v3 << 1, 2, 3;
    M3X m3 = m2.colwise() + v3;
    cout << "m3:\n" << m3 << endl;

}

void EigenArray()
{
    Eigen::Array22f a1;
    a1 << 1, 2, 3, 4;
    Eigen::Array22f a2;
    a2 << 3, 2, 1, 0;
    cout << a1 * a2 << endl;

    MXX m1(3, 2);
    m1 << 4, 5, 8, 3, 5, 0;
    cout << "m1:\n" << m1 << endl;
    cout << (m1.array() - 1.2).matrix() << endl;
}

/**
 * 2 Eigen 线性方程运算
 * http://zhaoxuhui.top/blog/2019/08/22/eigen-note-2.html
 */

void EigenLinearSolve()
{
    // 1 householder
    M3X A;
    V3X b;
    A << 1, 2, 3, 4, 5, 6, 7, 8, 10;
    b << 3, 3, 4;
    V3X x;
    x = A.colPivHouseholderQr().solve(b);
    cout << "1 householder " << x.transpose() << endl;

    // 2 LLT (cholesky)
    M2X A2, b2;
    A2 << 2, -1, -1, 3;
    b2 << 1, 2, 3, 1;
    M2X x2 = A2.ldlt().solve(b2);
    cout << "2 llt\n " << x2.transpose() << endl;
}

void EigenValues()
{
    M2X A;
    A << 1, 2, 2, 3;
    Eigen::EigenSolver<M2X> eigensolver(A);
    if (eigensolver.info() == Eigen::Success){
        cout << eigensolver.eigenvalues().transpose() << endl;
        cout << eigensolver.eigenvectors() << endl;
    }
    else{
        cout << "error while solving...";
    }
}


/**
 * (1)LU三角分解
三角分解法是仅对方阵有效，将原方阵分解成一个上三角形矩阵或是排列(permuted)的上三角形矩阵和一个下三角形矩阵，
这样的分解法又称为LU分解法。它的用途主要在简化一个大矩阵的行列式值的计算过程、求反矩阵和求解联立方程组。
不过要注意这种分解法所得到的上下三角形矩阵并非唯一，还可找到数个不同的一对上下三角形矩阵，
此两三角形矩阵相乘也会得到原矩阵。[L,U]=lu(A)

(2)QR分解
QR分解法对象不一定是方阵，其将矩阵分解成一个正规正交矩阵与上三角形矩阵,所以称为QR分解法,
与此正规正交矩阵的通用符号Q有关。[Q,R]=qr(A)

(3)SVD分解
奇异值分解(singular value decomposition,SVD)是另一种正交矩阵分解法；SVD是最可靠的分解法，
但是它比QR分解法要花上近十倍的计算时间。[U,S,V]=svd(A)，其中U和V分别代表两个正交矩阵，
而S代表一对角矩阵。和QR分解法相同，原矩阵A不必为方阵。使用SVD分解法的用途是解最小平方误差法和数据压缩。

(4)LLT分解
又称Cholesky分解，其把一个对称正定的矩阵表示成一个下三角矩阵L和其转置的乘积的分解。
它要求矩阵为方阵，且所有特征值必须大于零，故分解的下三角的对角元也是大于零的(LU三角分解法的变形)。A=LL^T

(5)LDLT分解
LDLT分解法实际上是Cholesky分解法的改进，因为Cholesky分解法虽然不需要选主元，但其运算过程中涉及到开方问题，
而LDLT分解法则避免了这一问题，可用于求解线性方程组。 也需要分解对象为方阵，分解结果为A=LDL^T。
其中L为一下三角形单位矩阵(即主对角线元素皆为1)，D为一对角矩阵(只在主对角线上有元素，其余皆为零)，
 L^T为L的转置矩阵。
 */
void EigenMatrixDecomposition()
{

}

/**
 * 最小二乘:SVD、QR、正规方程三种方法求解线性最小二乘问题
 * 其中SVD精度最高、速度最慢，
 * 正规方程法精度最低、速度最快，
 * QR方法则介于二者之间
 */
void Eigen_SVD_QR_LDLT()
{
    EigenMatrixDecomposition();

    // 1 SVD
    MXX A1(3, 2);
    V3X b1;
    V2X x1;
    A1 << 1, 2, 5, -3, 7, 10;
    b1 << 3, 7, 1;
    x1 = A1.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b1);
    cout <<"1 bdcSVD "<< x1.transpose() << endl;

    V3X error;
    double mean_error;
    error = (A1*x1 - b1).cwiseAbs();
    mean_error = error.mean();
    cout <<"check error: " << error.transpose() << endl << endl;
    cout << "check mean_error: " << mean_error << endl;

    // 2 QR
    x1 = A1.colPivHouseholderQr().solve(b1);
    cout << "2 QR " << x1.transpose() << endl << endl;

    // 3 ldlt
    /**
     * 正规方程法。所谓正规方程法思想是若要求解Ax=b，等价于方程两边同乘A的转置：ATAx=ATb
     * 需要注意的是如果系数矩阵A本身是病态的，采用正规方程法不是一个很好的选择。
     * 所谓病态矩阵是指矩阵A的行列式乘以A逆的行列式的结果，记为K，若这个值很大即为病态。
     */
    x1 = (A1.transpose()*A1).ldlt().solve(A1.transpose()*b1);
    cout << "3 ldlt " << x1.transpose() << endl << endl;
}

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


void initMats(SparseMatrix<EigenX> &A, VXX &b){
    A.insert(0, 0) = 2;
    A.insert(1, 1) = 1;
    A.insert(1, 2) = 2;
    A.insert(2, 2) = 1;
    A.insert(3, 3) = 1;
    A.insert(3, 5) = 1;
    A.insert(4, 2) = 1;
    A.insert(4, 3) = 1;
    A.insert(4, 4) = 1;
    A.insert(5, 5) = 1;

    b << 2, 4, 1, 5, 8, 3;
}

int EigenSparseSolve()
{
    // 新建矩阵，注意要指定大小
    SparseMatrix<EigenX> A(6,6);
    VXX b(6), x;

    // 为了代码简洁，将赋值代码单独写成了函数
    initMats(A, b);

    //构建solver
    SparseLU<SparseMatrix<EigenX>> solver;

    // compute步骤
    solver.compute(A);
    // solver步骤
    x = solver.solve(b);

    // 输出结果
    for (int i = 0; i < x.rows(); i++){
        cout << "x" << i << " = " << x(i) << endl;
    }
    return 1;
}

/**
| Module                                                       | HeaderFile                              | contents                                                     |
| ------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------ |
| SparseCore             | #include <Eigen/SparseCore>             | SparseMatrix and SparseVector classes, matrix assembly, basic sparse linear algebra (including sparse triangular solvers) |
| SparseCholesky         | #include <Eigen/SparseCholesky>         | Direct sparse LLT and LDLT Cholesky factorization to solve sparse self-adjoint positive definite problems |
| SparseLU               | #include<Eigen/SparseLU>                | Sparse LU factorization to solve general square sparse systems |
| SparseQR               | #include<Eigen/SparseQR>                | Sparse QR factorization for solving sparse linear least-squares problems |
| IterativeLinearSolvers | #include <Eigen/IterativeLinearSolvers> | Iterative solvers to solve large general linear square problems (including self-adjoint positive definite problems) |
| Sparse                 | #include <Eigen/Sparse>                 | Includes all the above modules                               |
*/
//http://zhaoxuhui.top/blog/2019/08/28/eigen-note-3.html
void EigenSparse()
{
    // 1 创建 SparseMatrix
    SparseMatrix<EigenX> m11(100, 50);
    SparseMatrix<int, RowMajor> m12(500, 1000);
    SparseMatrix<EigenX> m13;
    m13.resize(2000, 1000);
    SparseVector<EigenX> vec(1500);
    cout << "m1 row:" << m11.rows() << " col:" << m11.cols() << endl;
    cout << "m2 row:" << m12.rows() << " col:" << m12.cols() << endl;
    cout << "m3 row:" << m13.rows() << " col:" << m13.cols() << endl;
    cout << "vec row:" << vec.rows() << " col:" << vec.cols() << endl;

    // 2 初始化 SparseMatrix： 使用triplet
    // 2.1 triplet 的创建示例
    Triplet<EigenX> t1(185, 441, 2.6);
    cout << "row " << t1.row() << endl;
    cout << "col " << t1.col() << endl;
    cout << "val " << t1.value() << endl;

    // 2.2 初始化
    vector<Triplet<EigenX>> triplets;
    SparseMatrix<EigenX> sm(100, 100);

    // 构造数据
    for (int i = 1; i < 101; i++){
        triplets.push_back(Triplet<EigenX>(i - 1, i - 1, i));
    }

    // 两种不同方式设置数据
    // 方法1：将多个Triplet组成一个列表，调用稀疏矩阵的成员函数setFromTriplets()即可
    sm.setFromTriplets(triplets.begin(), triplets.end());
    // 方法2：调用稀疏矩阵的成员函数.insert()直接插入数值
    // 可以不用新建Triplet对象和列表，可能会有更高性能、内存占用更少
    sm.insert(5, 1) = 7;

    cout << sm.block(0, 0, 10, 10) << endl;

    // 3 SparseMatrix 的运算
    // 需要注意的是进行运算的稀疏矩阵的储存顺序必须要一致，否则在编译时就会出错。例如计算A^T+A，A^T必须与A有相同的存储顺序
    SparseMatrix<EigenX> sm1(2, 2);
    sm1.insert(0, 0) = 0;
    sm1.insert(0, 1) = 0;
    sm1.insert(1, 0) = 4;
    sm1.insert(1, 1) = 1;
    cout <<"sm1\n" << sm1 << endl;

    SparseMatrix<EigenX> sm2(2, 2);
    sm2.insert(0, 0) = 1;
    sm2.insert(0, 1) = 0;
    sm2.insert(1, 0) = 0;
    sm2.insert(1, 1) = 2;
    cout <<"sm2\n" << sm2 << endl;

    SparseMatrix<EigenX> sm3;
    sm3 = SparseMatrix<EigenX,ColMajor>(sm1.transpose()) + sm2;
    // sm3 = sm1.transpose() + sm2; // 会报错
    cout <<"sm3\n" << sm3 << endl;

    // 4 SparseMatrix 方程求解
    /**
        SparseMatrix<double> A;
        // fill A
        VectorXd b, x;
        // fill b
        // solve Ax = b
        SolverClassName<SparseMatrix<double> > solver;
        solver.compute(A);
        if(solver.info()!=Success) {
          // decomposition failed
          return;
        }
        x = solver.solve(b);
        if(solver.info()!=Success) {
          // solving failed
          return;
        }
        // solve for another right hand side:
        x1 = solver.solve(b1);

        以看到核心只有两步：compute()和solve()。在构造好求解器solver后，调用solver的compute()函数，传入稀疏矩阵A，
        然后调用solver的solve()函数，传入等号右边数字b，求解的结果由solve()函数返回。当然如果为了方便还可以将两步写在一行里，
        更为简洁x = solver.compute(A).solve(b);
     */
    /**
     * 前面说了需要注意各种方法的特点，在介绍各方法的特点之前先复习几个矩阵论中的概念。
        共轭矩阵(Self-adjoint Matrix):共轭矩阵，又称自共轭矩阵。共轭矩阵是矩阵本身先转置再把矩阵中每个元素取共轭得到的矩阵。矩阵中每一个第i行第j列的元素都与第j行第i列的元素的共轭相等。
        厄米特矩阵(Hermitian Matrix)：共轭矩阵的一种。在共轭矩阵的基础上，厄米特矩阵要求主对角线上的元素必须为实数。对于只包含实数元素的矩阵(实矩阵)，如果它是对称阵，即所有元素关于主对角线对称，那么它是厄米特矩阵。厄米特矩阵可看作是实对称矩阵的推广。

        各类方法特点及其适用的矩阵类型如下：
        LLT:用于self-adjoint matrices，SPD(Symmetric Positive Definite)，对称正定矩阵
        LDLT:用于general hermitian matrices，SPD(Symmetric Positive Definite)，对称正定矩阵
        LU:用于non hermitian matrices，必须为方阵
        QR:用于rectangular matrices，任意大小矩阵，适用于最小二乘问题

        所以对于代码中的方程组，其并不属于厄米特矩阵，所以LLT和LDLT方法都不能用。
        从这四个方法也可以看出来，QR方法是最通用的，其次是LU方法，最后是LLT和LDLT方法。
     */
    EigenSparseSolve();
}



void drawRectangle(Mat &img, Point2i p1, Point2i p2, Point2i p3, Point2i p4, Scalar color, int thickness) {
    line(img, p1, p2, color, thickness, LINE_AA);
    line(img, p2, p3, color, thickness, LINE_AA);
    line(img, p3, p4, color, thickness, LINE_AA);
    line(img, p4, p1, color, thickness, LINE_AA);
}

Point2i cvtVector2Point(V2X vec) {
    Point2i p(int(vec(0, 0)), int(vec(1, 0)));
    return p;
}

void EigenRotation2D()
{
    Point2i lp1(100, 100), lp2(300, 100), lp3(300, 300), lp4(100, 300);
    Mat background = Mat(400, 800, CV_8UC3, Scalar(0, 0, 0));

    Rotation2Df pose1(0);
    Rotation2Df pose2(3.141592653);
    V2X rp1(-100, -100), loc1;
    V2X rp2(100, -100), loc2;
    V2X rp3(100, 100), loc3;
    V2X rp4(-100, 100), loc4;
    V2X center(600, 200);
    M2X rot;

    int steps = 100;
    while (true) {
        for (int i = 0; i < steps; ++i) {
            float ratio = i * 1.0 / steps;
            // 注意这里插值的规则。取的是逆时针或顺时针旋转角最小的那个方向。
            // 例如如果让他内插0到270度，他会选择逆时针旋转90度而不是顺时针旋转270度。
            // 因此如果这里直接设置终止为2pi，他会直接选择逆时针旋转0度，结果就是没有旋转。
            // 所以插值的直接上下限不能超过180度，否则就可能和你预计的方向相反了。后面可以通过系数t则可以扩展上限，从而达到2pi。
            // 例如0-60 t设为6，0-90 t设为4，0-180 t设为2，这些都是等价的，且可以旋转360度
            rot = pose1.slerp(ratio * 2, pose2).toRotationMatrix();
            loc1 = rot * rp1 + center;
            loc2 = rot * rp2 + center;
            loc3 = rot * rp3 + center;
            loc4 = rot * rp4 + center;

            drawRectangle(background, lp1, lp2, lp3, lp4, Scalar(0, 0, 255), 2);
            drawRectangle(background,
                          cvtVector2Point(loc1),
                          cvtVector2Point(loc2),
                          cvtVector2Point(loc3),
                          cvtVector2Point(loc4),
                          Scalar(255, 255, 255), 2);
            circle(background, cvtVector2Point(loc1), 4, Scalar(255, 0, 0), -1, LINE_AA);
            circle(background, cvtVector2Point(loc2), 4, Scalar(0, 255, 0), -1, LINE_AA);
            circle(background, cvtVector2Point(loc3), 4, Scalar(0, 0, 255), -1, LINE_AA);
            circle(background, cvtVector2Point(loc4), 4, Scalar(0, 255, 255), -1, LINE_AA);
            circle(background, cvtVector2Point(V2X(0, 0)), 4, Scalar(0, 0, 255), -1, LINE_AA);

            imshow("img", background);
            waitKey(50);

            background = Scalar(0, 0, 0);
        }
    }
}

/**
 *
Eigen::Matrix3d      //旋转矩阵（3*3）
Eigen::AngleAxisd    //旋转向量（3*1）
Eigen::Vector3d      //欧拉角（3*1）
Eigen::Quaterniond   //四元数（4*1）
Eigen::Isometry3d    //欧式变换矩阵（4*4） 等距变换(Isometry
Eigen::Affine3d      //放射变换矩阵（4*4）
Eigen::Projective3d  //射影变换矩阵（4*4）
 */
 /**
  * 这个变换顺序(先平移后旋转or先旋转后平移)是什么？
  * 这个变换相对谁改变(固定轴or自身)？
  */
void TestEigenGeometry() {
    //旋转向量使用AngleAxis，运算可以当做矩阵
    AxisX rotation_vector(M_PI / 2, V3X(0,0,1));     //眼Z轴旋转45°
//    V3X translation(0,0,0);
    V3X translation(1, 3, 4);
    cout << "rotation matrix = \n" << rotation_vector.matrix() << endl;

    //欧式变换矩阵使用Eigen::Isometry
    /**
     * 不初始化为单位阵行不行？
     * 也可以，那就不要调用.translate()和.rotate()函数，
     * 直接调用.matrix()函数获取对应的变换矩阵，然后手动自己设置(构造)
     */
    ISO3X tWC = ISO3X::Identity();      //实质为4*4的矩阵
    tWC.rotate(rotation_vector);        //按照rotation_vector进行转化
    tWC.pretranslate(translation);      //平移向量设为translation
    cout << "tWC = \n" << tWC.matrix() << endl;

    V4X rC(1,0,0, 1); // vector in camera
    cout << "rC before = \n" << rC.transpose() <<endl;
    cout << "rC -> rW = \n" << (tWC * rC).transpose() << endl;
    cout << "rC -> rW = \n" << (tWC.matrix() * rC).transpose() << endl;

    V4X rw(1,0,0, 1); // vector in camera
    cout << "rw before = \n" << rC.transpose() <<endl;
    cout << "rw -> rC = \n" << (tWC.inverse() * rC).transpose() << endl;
}

void EigenGeometry()
{
//    EigenRotation2D();
    TestEigenGeometry();
}


void TestEigen()
{
    // 1 Eigen 矩阵运算
//    EigenArray();
//    EigenBroadcast();
//    EigenMap();
//    EigenOther();

    // 2 Eigen 线性方程解算
//    EigenLinearSolve();
//    Eigen_SVD_QR_LDLT();
//    QR();
//    EigenValues();

    // 3 Eigen 稀疏矩阵
//    EigenSparse();

    // 4 Eigen 几何变换：旋转、平移
    EigenGeometry();


//    ImuInt();

}




