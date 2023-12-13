/*
 * @Author: https://github.com/haohaoalt
 * @Date: 2023-12-13 13:50:59
 * @LastEditors: hayden haohaoalt@163.com
 * @LastEditTime: 2023-12-13 14:12:43
 * @FilePath: /hao_slambook2/ch3/useEigen/eigenMatrix.cpp
 * @Description: 
 * Copyright (c) 2023 by haohaoalt@163.com, All Rights Reserved. 
 */
#include<iostream>
using namespace std;
#include<ctime>
// Eigen core part
#include <Eigen/Core>
// dense matrix computing
#include <Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc, char ** argv)
{
     //hayden: <type column row>
    Matrix<float,2,3> matrix_23;
    Vector3d v_3d;
    Matrix<float,3,1> vd_3d;

    Matrix3d matrix_33 = Matrix3d::Zero();
    // unknown matrix size
    Matrix<double,Dynamic,Dynamic> matrix_dynamic;
    MatrixXd matrix_x;
    
    //下面是对eigen阵的操作
    matrix_23 << 1,2,3,4,5,6;
    cout << "matrix 2x3 from 1 to 6: " << endl;
    cout << matrix_23 << endl;

    printf("------------------------------------- \n");
    cout << "print matrix 2x3: " << endl;
    for(int i = 0; i < 2; i ++)
    {
        for(int j = 0; j < 3; j++)
        {
            cout << matrix_23(i,j) << "\t";
        }
        cout << endl;
    }
    //矩阵和向量相乘
    v_3d << 3,2,1;
    vd_3d << 4,5,6;

    Matrix<double,2,1> result1 = matrix_23.cast<double>() * v_3d;
    printf("------------------------------------- \n");
    cout << "[1,2,3;4,5,6]*[3,2,1]: " << result1.transpose() << endl;
    printf("------------------------------------- \n");
    Matrix<float,2,1> result2 = matrix_23 * vd_3d;
    cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

    matrix_33 = Matrix3d::Random();
    cout << "random matrix : \n" << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl;
    cout << "sum :" << matrix_33.sum() << endl;
    cout << "trance: " << matrix_33.trace() << endl;
    cout << "time 10: " << 10*matrix_33 << endl;
    cout << "inverse: \n" << matrix_33.inverse() << endl;
    cout << "det: " << matrix_33.determinant() << endl;
    // 特征值
    // 实对称矩阵可以保证对角化成功
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n"
         << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n"
         << eigen_solver.eigenvectors() << endl;

    // 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N的大小在前边的宏里定义，它由随机数生成
    // 直接求逆自然是最直接的，但是求逆运算量大

    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); // 保证半正定
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock(); // 计时
    // 直接求逆
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // 通常用矩阵分解来求，例如QR分解，速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // 对于正定矩阵，还可以用cholesky分解来解方程
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
         << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;
}

