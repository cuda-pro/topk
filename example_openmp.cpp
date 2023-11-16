// g++ -fopenmp example_openmp.cpp -o bin/example_openmp --std=c++11 -O3
/*
- https://lemon-412.github.io/imgs/20200516OpenMP_simple_Program.pdf
- https://scc.ustc.edu.cn/zlsc/cxyy/200910/W020121113517997951933.pdf
- https://openmpusers.org/wp-content/uploads/uk-openmp-users-2018-OpenMP45Tutorial_new.pdf
- https://enccs.github.io/openmp-gpu/
- https://www.openmp.org/resources/openmp-books/

please Think for yourself
please Think for yourself
please Think for yourself
*/

#include <omp.h>

#include <chrono>
#include <iostream>
#include <vector>
using namespace std;

int g_ncore = omp_get_num_procs();  // 获取执行核的数量

int dtn(int n, int min_n) {
    int max_tn = n / min_n;
    int tn = max_tn > g_ncore ? g_ncore : max_tn;  // tn 表示要设置的线程数量
    if (tn < 1) {
        tn = 1;
    }
    return tn;
}

void Matrix_Multiply(int *a, int row_a, int col_a,
                     int *b, int row_b, int col_b,
                     int *c, int c_size) {
    if (col_a != row_b || c_size < row_a * col_b) {
        return;
    }
    int i, j, k;
    // #pragma omp for private(i, j, k)
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
    if (col_a != row_b || c_size < row_a * col_b) {
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
    std::vector<int> a(1000000);
    int start_doc_id = 10;
    // omp_set_num_threads(2);
#pragma omp parallel for
    for (int i = 0; i < 10000000; i++) {
        // a[i] = i + start_doc_id;
        //  cout << omp_get_thread_num() << endl;
    }
    std::cout << a.size() << std::endl;
}

void case2() {
    int i, j;
    int a[100][100] = {0};
#pragma omp parallel for schedule(static)
    for (i = 0; i < 100; i++) {
        for (j = i; j < 100; j++) {
            a[i][j] = i * j;
            // printf("a[%d][%d]=%d, tn=%d\n", i, j, a[i][j], omp_get_thread_num());
        }
    }
}

void case3() {
    int a[2][3] = {
        {1, 2, 3},
        {1, 2, 3},
    };
    int b[3][2] = {
        {1, 2},
        {1, 2},
        {1, 2},
    };
    int *c = new int[2 * 2];

#ifdef PARALLEL_MATRIX_MULTIPLY
    Parallel_Matrix_Multiply(a[0], 2, 3, b[0], 3, 2, c, 4);
#else
    Matrix_Multiply(a[0], 2, 3, b[0], 3, 2, c, 4);
#endif

    for (int i = 0; i < 4; i++) {
        printf("%d,", c[i]);
    }

    free(c);
}

#define N 100000000

int fun() {
    int a = 0;
    for (int i = 0; i < N; ++i) {
        a += i;
    }
    return a;
}
void case4() {
#pragma omp parallel for num_threads(2) schedule(dynamic)
    for (int i = 0; i < 100; ++i) {
        fun();
    }
    std::cout << "finish" << std::endl;
}

void test() {
    int a = 0;
    double t1, t2;
    t1 = omp_get_wtime();
    for (int i = 0; i < 100000000; i++) {
        a = i + 1;
    }
    t2 = omp_get_wtime();
    printf("Time = %f s\n", t2 - t1);
}
void case5() {
    double t1, t2;
    t1 = omp_get_wtime();
#pragma omp parallel for
    for (int j = 0; j < 2; j++) {
        test();
    }
    t2 = omp_get_wtime();
    printf("Total time = %f s\n", t2 - t1);
    test();
}

/* Seriel Code */
static long num_steps = 100000;
double step;
void pi_serial() {
    int i;
    double x, pi, sum = 0.0, start_time, end_time;
    step = 1.0 / (double)num_steps;
    // start_time = clock();
    start_time = omp_get_wtime();
    for (i = 1; i <= num_steps; i++) {
        x = (i - 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    // end_time = clock();
    end_time = omp_get_wtime();
    printf("Pi = %f\t pi_serial Running time %f s\n", pi, end_time - start_time);
}

#define NUM_THREADS 4
void pi_parallel() {
    int i;
    double pi, sum[NUM_THREADS], start_time, end_time;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    // start_time = clock();
    start_time = omp_get_wtime();
#pragma omp parallel
    {
        int id;
        double x;
        id = omp_get_thread_num();
        for (i = id, sum[id] = 0.0; i < num_steps; i = i + NUM_THREADS) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++) pi += sum[i] * step;
    // end_time = clock();
    end_time = omp_get_wtime();
    printf("Pi = %f\t parallel Running time %f s\n", pi, end_time - start_time);
}

void pi_parallel_for() {
    int i, id;
    double x, pi, sum[NUM_THREADS], start_time, end_time;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    // start_time = clock();
    start_time = omp_get_wtime();
#pragma omp parallel private(x, id)
    {
        id = omp_get_thread_num();
        sum[id] = 0;
#pragma omp for
        for (i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum[id] += 4.0 / (1.0 + x * x);
        }
    }
    for (i = 0, pi = 0.0; i < NUM_THREADS; i++) pi += sum[i] * step;
    // end_time = clock();
    end_time = omp_get_wtime();
    printf("Pi = %f\t parallel for Running time %f s\n", pi, end_time - start_time);
}

int main() {
    case5();
    pi_serial();
    pi_parallel();
    pi_parallel_for();

    return 0;
}
