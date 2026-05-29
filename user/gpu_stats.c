#include "user.h"
#include "fcntl.h"
#include "gpu.h"



int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    int fd = open("/gpu", O_RDWR);
    if (fd < 0) {
        fprintf(2, "gpu_stats: cannot open /gpu\n");
        exit(1);
    }

    struct gpu_stats st;
    if (ioctl(fd, GPU_IOC_GET_STATS, (uint64)&st) < 0) {
        fprintf(2, "gpu_stats: GPU_IOC_GET_STATS failed\n");
        close(fd);
        exit(1);
    }

    printf("gpu matmul_ops: %llu\n", (unsigned long long)st.matmul_ops);

    close(fd);
    exit(0);
}
