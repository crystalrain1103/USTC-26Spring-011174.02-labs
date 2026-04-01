#include "user.h"
#include "fcntl.h"
#include "gpu.h"

int main() {
    int fd = open("/gpu", O_RDWR);
    if (fd < 0) {
        printf("[gpu_stats] open failed\n");
        exit(1);
    }
    struct gpu_stats st;
    ioctl(fd, GPU_IOC_GET_STATS, (uint64)&st);
    printf("gpu matmul_ops: %llu", st.matmul_ops);
    close(fd);
    exit(0);
}