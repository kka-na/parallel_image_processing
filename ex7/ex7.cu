#include <stdio.h>
#include </usr/local/cuda-11.4/include/cuda_runtime.h>
#include </usr/local/cuda-11.4/samples/common/inc/helper_timer.h>
#include </usr/local/cuda-11.4/samples/common/inc/helper_cuda.h>

int timer_test(void)
{
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    int a = 0;
    for (int i = 0; i < 100; i++)
    {
        a += 1;
    }
    sdkStopTimer(&timer);
    double time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    printf("Processing Time %fsec\n", time);
    return 0;
}