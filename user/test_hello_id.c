#include "user.h"

int main(void) {
    int tag = 12345;  // TODO: 可以改成你自己的学号后几位
    int ret = hello_id(tag);
    printf("hello_id(%d) returned %d\n", tag, ret);
    exit(0);
}