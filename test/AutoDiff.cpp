//
// Created by aqiu on 2021/1/27.
//

#include "AutoDiff.h"

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
using namespace std;

/*############################### 数值微分 begin ################################*/
// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

};

struct my_functor : Functor<double>
{
    my_functor(void): Functor<double>(2,2) {}
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        // Implement y = 10*(x0+3)^2 + (x1-5)^2
        fvec(0) = 10.0*pow(x(0)+3.0,2) +  pow(x(1)-5.0,2);
        fvec(1) = 0;

        return 0;
    }
};

/*############################### 数值微分 end ################################*/


template<typename Input, typename Output>
void cost(const Eigen::MatrixBase<Input> &x, Output &y)
{
    Eigen::Matrix2d Q;
    Q << 2, 1,
         1, 4;

    // y = 0.5*x.dot(Q * x); // does not work for second order derivatives

    using ScalarT = typename Eigen::MatrixBase<Input>::Scalar;
    y = 0.5*x.dot(Q.template cast<ScalarT>() * x);
}

void AutoDiff_gradient(void)
{
    std::cout << "\n" << __FUNCTION__ << std::endl;

    const int NX = 2;
    using Scalar = double;

    /* first order derivative */
    using Derivatives = Eigen::Matrix<Scalar, NX, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    using ADx_t = Eigen::Matrix<ADScalar, NX, 1>;

    ADx_t x = {1, 2};
    /* initialize derivatives */
    for (int i=0; i<x.rows(); i++) {
        x[i].derivatives().coeffRef(i) = 1;
    }

    ADScalar y;
    cost(x, y);

    std::cout << "input: x = " << x.transpose() << std::endl;
    std::cout << "output: y = " << y << std::endl;
    std::cout << "gradient: dy = " << y.derivatives().transpose() << std::endl;
}

void AutoDiff_gradient_hessian(void)
{
    std::cout << "\n" << __FUNCTION__ << std::endl;

    const int NX = 2;
    using Scalar = double;

    /* second order derivative */
    using Derivatives = Eigen::Matrix<Scalar, NX, 1>;
    using ADScalar = Eigen::AutoDiffScalar<Derivatives>;
    using outerDerivatives = Eigen::Matrix<ADScalar, NX, 1>;
    using outerADScalar = Eigen::AutoDiffScalar<outerDerivatives>;
    using Hessian_t = Eigen::Matrix<Scalar, NX, NX>;
    using ADx_t = Eigen::Matrix<outerADScalar, NX, 1>;

    ADx_t x = {1, 2};

    /* initialize derivatives */
    for (int i = 0; i < NX; i++) {
        x(i).value().derivatives() = Derivatives::Unit(NX, i);
        x(i).derivatives() = Derivatives::Unit(NX, i);
        /* initialize hessian matrix to zero */
        for (int j = 0; j < NX; j++) {
            x(i).derivatives()(j).derivatives().setZero();
        }
    }

    outerADScalar y;
    cost(x, y);

    Scalar val;
    Derivatives grad;
    Hessian_t hess;

    val = y.value().value();
    grad = y.value().derivatives();

    /* extract hessian */
    for (int i = 0; i < NX; i++) {
        hess.template middleRows<1>(i) = y.derivatives()(i).derivatives().transpose();
    }

    std::cout << "input: x " << x.transpose() << std::endl;
    std::cout << "output: y " << y << std::endl;
    std::cout << "value " << val << std::endl;
    std::cout << "gradient " << grad.transpose() << std::endl;
    std::cout << "hessian \n" << hess << std::endl;
}

template<typename T>
T scalarFunctionOne(T const & x) {
    return 2*x*x + 3*x + 1;
};

void checkFunctionOne(double & x, double & dfdx) {
    dfdx = 4*x + 3;
}

template<typename T>
T scalarFunctionTwo(T const & x, T const & y) {
    return 2*x*x + 3*x + 3*x*y*y + 2*y + 1;
};

void checkFunctionTwo(double & x, double & y, double & dfdx, double & dfdy ) {
    dfdx = 4*x + 3 + 3*y*y;
    dfdy = 6*x*y + 2;
}

//-----------------------------------------------------------
//https://github.com/libigl/eigen/blob/master/unsupported/test/autodiff.cpp
void TestAutoDiffGood()
{
    std::cout << "_______________ [TestAutoDiffGood] ______________ " << std::endl;
}
// https://gist.github.com/nuft/828bd48994a1f18e85b806e8e417b6d4
void TestAutoDiff1()
{
    std::cout << "_______________ [TestAutoDiff1] ______________ " << std::endl;
    AutoDiff_gradient();
    AutoDiff_gradient_hessian();
}

//https://joelcfd.com/automatic-differentiation/
void TestAutoDiff2()
{
    std::cout << "_______________ [TestAutoDiff2] ______________ " << std::endl;
    double x, y, z, f, g, dfdx, dgdy, dgdz;
    Eigen::AutoDiffScalar<Eigen::VectorXd> xA, yA, zA, fA, gA;

    cout << endl << "Testing scalar function with 1 input..." << endl;
    xA.value() = 1;
    xA.derivatives() = Eigen::VectorXd::Unit(1, 0);
    fA = scalarFunctionOne(xA);
    cout << "  AutoDiff:" << endl;
    cout << "    Function output: " << fA.value() << endl;
    cout << "    Derivative: " << fA.derivatives() << endl;

    x = 1;
    checkFunctionOne(x, dfdx);
    cout << "  Hand differentiation:" << endl;
    cout << "    Derivative: " << dfdx << endl << endl;


    cout << "Testing scalar function with 2 inputs..." << endl;
    yA.value() = 1;
    zA.value() = 2;
    yA.derivatives() = Eigen::VectorXd::Unit(2, 0);
    zA.derivatives() = Eigen::VectorXd::Unit(2, 1);
    gA = scalarFunctionTwo(yA, zA);
    cout << "  AutoDiff:" << endl;
    cout << "    Function output: " << gA.value() << endl;
    cout << "    Derivative: " << gA.derivatives()[0] << ", " << gA.derivatives()[1] << endl;

    y = 1;
    z = 2;
    checkFunctionTwo(y, z, dgdy, dgdz);
    cout << "  Hand differentiation:" << endl;
    cout << "    Derivative: " << dgdy << ", " << dgdz << endl;
}

//https://github.com/libigl/eigen/blob/master/unsupported/test/levenberg_marquardt.cpp
//https://gitlab.uliege.be/C.Duchesne/trio/blob/236a9b7f6c6c66cfeee52cc2241aaed7ff760b69/eigen/unsupported/test/levenberg_marquardt.cpp
void TestAutoDiffOfLM()
{
    std::cout << "_______________ [TestAutoDiffOfLM] ______________ " << std::endl;

}

//https://stackoverflow.com/questions/18509228/how-to-use-the-eigen-unsupported-levenberg-marquardt-implementation
void TestAutoDiffOfLM1()
{
    std::cout << "_______________ [TestAutoDiffOfLM1] ______________ " << std::endl;
    Eigen::VectorXd x(2);
    x(0) = 2.0;
    x(1) = 3.0;
    std::cout << "x: " << x << std::endl;

    my_functor functor;
    Eigen::NumericalDiff<my_functor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>,double> lm(numDiff);
    lm.parameters.maxfev = 2000;
    lm.parameters.xtol = 1.0e-10;
    std::cout << lm.parameters.maxfev << std::endl;

    int ret = lm.minimize(x);
    std::cout << lm.iter << std::endl;
    std::cout << ret << std::endl;

    std::cout << "x that minimizes the function: " << x << std::endl;
    // std::cin.get();
}


void TestAutoDiff()
{
    TestAutoDiffGood();
    TestAutoDiff1();
    TestAutoDiff2();
    TestAutoDiffOfLM();
    TestAutoDiffOfLM1();
}