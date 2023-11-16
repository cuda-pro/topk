// g++ -fopenmp example_openmp.cpp -o bin/example_openmp --std=c++11 -O3

#include <omp.h>

#include <iostream>
#include <vector>
using namespace std;

int g_ncore = omp_get_num_procs();  // 获取执行核的数量

/** 计算循环迭代需要的线程数量
根据循环迭代次数和 CPU 核数及一个线程最少需要的循环迭代次数
来计算出需要的线程数量，计算出的最大线程数量丌超过 CPU 核数
@param int n - 循环迭代次数
@param int min_n - 单个线程需要的最少迭代次数
@return int - 线程数量
*/
int dtn(int n, int min_n) {
    int max_tn = n / min_n;
    int tn = max_tn > g_ncore ? g_ncore : max_tn;  // tn 表示要设置的线程数量
    if (tn < 1) {
        tn = 1;
    }
    return tn;
}

/** 矩阵串行乘法函数
@param int *a - 指向要相乘的第个矩阵的指针
@param int row_a - 矩阵 a 的行数
@param int col_a - 矩阵 a 的列数
@param int *b - 指向要相乘的第个矩阵的指针
@param int row_b - 矩阵 b 的行数
@param int col_b - 矩阵 b 的列数
@param int *c - 计算结果的矩阵的指针
@param int c_size - 矩阵 c 的空间大小（总元素个数）
@return void - 无
*/
void Matrix_Multiply(int *a, int row_a, int col_a,
                     int *b, int row_b, int col_b,
                     int *c, int c_size) {
    if (col_a != row_b || c_size < row_a * col_b) {
        return;
    }
    int i, j, k;
#pragma omp for private(i, j, k)
    for (i = 0; i < row_a; i++) {
        int row_i = i * col_a;
        int row_c = i * col_b;
        for (j = 0; j < col_b; j++) {
            c[row_c + j] = 0;
            for (k = 0; k < row_b; k++) {
                c[row_c + j] += a[row_i + k] * b[k * col_b + j];
            }
        }
    }
}

void Parallel_Matrix_Multiply(int *a, int row_a, int col_a,
                              int *b, int row_b, int col_b,
                              int *c, int c_size) {
    if (col_a != row_b) {
        return;
    }
    int i, j, k;
    int index;
    int border = row_a * col_b;
    i = 0;
    j = 0;
#pragma omp parallel private(i, j, k) num_threads(dtn(border, 1))
    for (index = 0; index < border; index++) {
        i = index / col_b;
        j = index % col_b;
        int row_i = i * col_a;
        int row_c = i * col_b;
        c[row_c + j] = 0;
        for (k = 0; k < row_b; k++) {
            c[row_c + j] += a[row_i + k] * b[k * col_b + j];
        }
    }
}

void case1() {
    std::vector<int> a(10);
    int start_doc_id = 10;
    omp_set_num_threads(2);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < a.size(); i++) {
        a[i] = i + start_doc_id;
        cout << omp_get_thread_num() << endl;
    }

    for (auto i : a) {
        cout << i << ",";
    }
    cout << endl;
}

void case2() {
    int i, j;
    int a[100][100] = {0};
#pragma omp parallel for schedule(dynamic)
    for (i = 0; i < 100; i++) {
        for (j = i; j < 100; j++) {
            a[i][j] = i * j;
            printf("a[%d][%d]=%d, tn=%d\n", i, j, a[i][j], omp_get_thread_num());
        }
    }
}

int main() {
    case2();

    return 0;
}
