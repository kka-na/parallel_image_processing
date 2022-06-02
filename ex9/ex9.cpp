#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"

#include "CL/cl.hpp"

#include <chrono>

using namespace cv;

char *snuclLoadFile(const char *filename, size_t *length_ret);

int main()
{
    Mat src = imread("../image/Grab_Image.bmp", 0);
    resize(src, src, Size(512, 512));
    Mat dst_cl(src.size(), src.type());
    Mat dst_cv(src.size(), src.type());

    // OCL init
    int xorder = 0;
    int yorder = 1;
    cl_platform_id platform;
    cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
    cl_device_id device;
    cl_context context;
    cl_command_queue cmq;
    cl_program program;
    cl_kernel kernel;
    cl_mem memSrc, memDst;
    cl_int err;
    cl_image_format clImageFormat;
    clImageFormat.image_channel_order = CL_A;
    clImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_sampler sampler;
    size_t size;
    size_t global[2], local[2];
    size_t origin[3] = {0, 0, 0};
    int i, j;
    size_t region[3] = {src.rows, src.cols, 1};
    //

    // OpenCL program
    auto s1 = std::chrono::high_resolution_clock::now();
    //// Step 1
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, dev_type, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cmq = clCreateCommandQueue(context, device, 0, NULL);

    //// Step 2
    size_t program_src_len;
    char *program_src = snuclLoadFile("../Device.cl", &program_src_len);
    program = clCreateProgramWithSource(context, 1, (const char **)&program_src, &program_src_len, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "SobelFilter3x3Image", NULL);

    //// Step 3
    global[0] = src.rows;
    global[1] = src.cols;
    local[0] = 32;
    local[1] = 8;
    size = global[0] * global[1] * sizeof(unsigned char);
    memSrc = clCreateImage2D(context, CL_MEM_READ_WRITE,
                             &clImageFormat,
                             src.rows,
                             src.cols,
                             0, 0, NULL);
    memDst = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, NULL);

    //// Step 4
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memSrc);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memDst);
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&src.rows);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&src.cols);
    err = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&xorder);
    err = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&yorder);

    // Step 5

    err = clEnqueueWriteImage(cmq, memSrc, CL_FALSE, origin, region, src.cols, 0, src.data, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(cmq, kernel, 2, NULL, global, local, 0, NULL, NULL);
    err = clEnqueueReadBuffer(cmq, memDst, CL_FALSE, 0, size, dst_cl.data, 0, NULL, NULL);

    auto f1 = std::chrono::high_resolution_clock::now();
    double d1 = double(std::chrono::duration_cast<std::chrono::microseconds>(f1 - s1).count()) / 1000000;
    printf("OpenCL Processing Time:  %1f msec \n", d1);

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(memSrc);
    clReleaseMemObject(memDst);

    // OpenCV
    auto s2 = std::chrono::high_resolution_clock::now();
    Sobel(src, dst_cv, -1, xorder, yorder, 3, 1, 0, BORDER_DEFAULT);
    auto f2 = std::chrono::high_resolution_clock::now();
    double d2 = double(std::chrono::duration_cast<std::chrono::microseconds>(f2 - s2).count()) / 1000000;
    printf("OpenCV Processing Time:  %1f msec \n", d2);

    imshow("OpenCL", dst_cl);
    imshow("OpenCV", dst_cv);

    imwrite("../result/OpenCL.jpg", dst_cl);
    imwrite("../result/OpenCV.jpg", dst_cv);

    waitKey(0);
    return 0;
}

char *snuclLoadFile(const char *filename, size_t *length_ret)
{
    FILE *fp;
    size_t length;
    fp = fopen(filename, "rb");
    if (fp == 0)
        return NULL;

    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *ret = (char *)malloc(length + 1);
    if (fread(ret, length, 1, fp) != 1)
    {
        fclose(fp);
        free(ret);
        return NULL;
    }

    fclose(fp);
    if (length_ret)
        *length_ret = length;
    ret[length] = '\0';

    return ret;
}
