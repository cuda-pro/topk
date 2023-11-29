// g++  -O3 -Wall --std=c++11 -march=native example_mmap_file.cpp -o bin/example_mmap_file
// g++  -O3 -Wall --std=c++11 -march=native example_mmap_file.cpp -o bin/example_mmap_file -DMULTI_PROCESS_MMAP_W_FILE
/*
todo:
- use cpu + mmap to generate random dataset file
- use gpu + mmap to generate random dataset file
*/

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
// #include <string>
// #include <vector>

using namespace std;

const int PROCESS_COUNT = 8;
const int RESULT_SIZE = 20000;
// const int DOC_SIZE = 128;

// just a example to write simple char[RESULT_SIZE * 2] 0\n1\n....
void work(const string &file_name, int id) {
    int fd = open(file_name.c_str(), O_RDWR | O_CREAT, 0666);
    char *buffer = (char *)mmap(NULL, RESULT_SIZE * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    int start = id * RESULT_SIZE / PROCESS_COUNT;
    int end = start + RESULT_SIZE / PROCESS_COUNT;
    for (int i = start; i < end; ++i) {
        buffer[i << 1] = (i % 2 != 0 ? '1' : '0');
        buffer[i << 1 | 1] = '\n';
    }

    munmap(buffer, RESULT_SIZE * 2);
}

void multi_process_mmap_w_file() {
    // main process init seek to last-1, write " "
    string predict_file = "result.txt";
    int fd = open(predict_file.c_str(), O_RDWR | O_CREAT, 0666);
    lseek(fd, RESULT_SIZE * 2 - 1, SEEK_SET);
    write(fd, " ", 1);
    close(fd);

    int id;
    pid_t pid;
    vector<pid_t> pids;
    for (int i = 1; i < PROCESS_COUNT; i++) {
        id = i;
        pid = fork();
        if (pid <= 0) break;
        pids.push_back(pid);
    }

    if (pid == -1) {
        cerr << "startup process failed" << endl;
    } else {
        if (pid == 0) {
            work(predict_file, id);  // subprocess
            exit(0);
        } else {
            work(predict_file, 0);  // main process
        }
    }

    exit(0);
}

void multi_process_mmap_copy(const char *dst, const char *src, int process_num) {
    int fd_src = open(src, O_RDONLY);
    if (fd_src < 0) {
        perror("open");
        exit(2);
    }

    // open dst file with create and truncate for w/r
    int fd_dst = open(dst, O_RDWR | O_CREAT | O_TRUNC, 0664);
    if (fd_dst < 0) {
        perror("open");
        exit(3);
    }

    struct stat sbuf;
    int ret = fstat(fd_src, &sbuf);
    if (ret < 0) {
        perror("fstat");
        exit(4);
    }
    int flen = sbuf.st_size;
    int n = process_num;
    if (flen < n) {
        n = flen;
    }

    // truncate dst file with src file size
    ret = ftruncate(fd_dst, flen);
    if (ret < 0) {
        perror("ftruncate");
        exit(5);
    }

    // shared memory to read from src file, start at 0
    char *mp_src = (char *)mmap(NULL, flen, PROT_READ, MAP_SHARED, fd_src, 0);
    if (mp_src == MAP_FAILED) {
        perror("mmap");
        exit(6);
    }
    close(fd_src);

    char *mp_dst = (char *)mmap(NULL, flen, PROT_READ | PROT_WRITE, MAP_SHARED, fd_dst, 0);
    if (mp_dst == MAP_FAILED) {
        perror("mmap");
        exit(7);
    }
    close(fd_dst);

    int bs = flen / n;
    int mod = flen % bs;
    char *temp_src = mp_src;
    char *temp_dst = mp_dst;

    int i;
    pid_t pid;
    for (i = 0; i < n; ++i) {
        printf("create %dth proc\n", i);
        if ((pid = fork()) == 0)
            break;
    }

    if (n == i) {  // main process
        int j = 0;
        for (j = 0; j < n; ++j)
            wait(NULL);
    } else if (i == (n - 1)) {
        printf("i = %d\n", i);
        memcpy(temp_dst + i * bs, temp_src + i * bs, bs + mod);
    } else {
        printf("i = %d\n", i);
        memcpy(temp_dst + i * bs, temp_src + i * bs, bs);
    }

    munmap(mp_src, flen);
    munmap(mp_dst, flen);
}

int main(int argc, char *argv[]) {
#ifdef MULTI_PROCESS_MMAP_W_FILE
    multi_process_mmap_w_file();
    return 0;
#endif

    int n;
    if (argc < 3 || argc > 4) {
        printf("Enter like this : bin file_src file_dst [proc_number=5]\n");
        exit(1);
    } else if (argc == 3)
        n = 5;
    else
        n = atoi(argv[3]);

    multi_process_mmap_copy(argv[2], argv[1], n);
    return 0;
}