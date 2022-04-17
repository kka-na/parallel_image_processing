#include <cv.h>
#include <highgui.h>
#include <omp.h>
#include <tchar.h>
#include <time.h>
#include <math.h>
#include <cxcore.h>
#include <iostream>

#define _nx 3
#define _ny 3
#define iteration 10

char *model = "serial";
int wf = 4;

void wInter(int x, int y, char *wf, float *w);
void Interp(unsigned char *src, int h, int width, char *wf, float *w, int x, int y, unsigned char *output);
void Interp_omp(unsigned char *src, int h, int width, char *wf, float *w, int x, int y, unsigned char *output);

using namespace std;

int main()
{

	int nx = _nx;
	int ny = _ny;

	IplImage *src = cvLoadImage("(Color_512)lena.bmp", CV_LOAD_IMAGE_GRAYSCALE); // load image using opencv
	int wd = src->width;
	int hg = src->height;
	int width = src->width;
	int height = src->height;

	IplImage *dst;
	dst = cvCreateImage(cvSize(wd * nx, hg * ny), 8, 1);

	int row, col;

	float *input = new float[wd * hg];
	memset(input, 0, wd * hg * sizeof(float));

	float *output = new float[(wd * nx) * (hg * ny)];
	memset(output, 0, (wd * nx) * (hg * ny) * sizeof(float));

	for (row = 0; row < hg; row++)
		for (col = 0; col < wd; col++)
			input[row * wd + col] = (float)cvGetReal2D(src, row, col);

	int iter = 0;
	int64 tStart, tEnd;
	float pTime;
	float t_min = 0, t_max = 0;
	float t_ave = 0;

	float *w = new float[nx * 8];
	memset(w, 0, nx * 8 * sizeof(float));

	for (iter = 0; iter < iteration; iter++)
	{
		printf("iteration number : %d. ", iter + 1);

		tStart = cvGetTickCount();

		wInter(nx, ny, "Bilinear", w);
		Interp((unsigned char *)src->imageData, src->height, src->width, "Bilinear", w, nx, ny, (unsigned char *)dst->imageData);
		// Interp_omp((unsigned char*)src->imageData, src->height, src->width, "Bilinear",w,nx, ny,(unsigned char*)dst->imageData);

		tEnd = cvGetTickCount();
		pTime = 0.001 * (tEnd - tStart) / cvGetTickFrequency();
		t_ave += pTime;

		printf("processing time : %.3f ms\n", pTime);
		if (iter == 0)
		{
			t_min = pTime;
			t_max = pTime;
		}
		else
		{
			if (pTime < t_min)
				t_min = pTime;
			if (pTime > t_max)
				t_max = pTime;
		}
	}

	if (iteration == 1)
		t_ave = t_ave;
	else if (iteration == 2)
		t_ave = t_ave / 2;
	else
		t_ave = (t_ave - t_min - t_max) / (iteration - 2);

	printf("\nAverage processing time : %.3f ms\n", t_ave);

	cvNamedWindow("src", 0);
	cvShowImage("src", src);

	cvNamedWindow("dst", 0);
	cvShowImage("dst", dst);

	cvWaitKey(0);

	cvSaveImage("result.bmp", dst);
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	delete[] input;
	delete[] output;

	return 0;
}

void wInter(int x, int y, char *wf, float *w)
{
	x = x - 1;
	y = y - 1;
	int i;

	for (i = 0; i < x; i++)
	{
		w[i * 2 + 0] = 1 - (float)(i + 1) / (float)(x + 1);
		w[i * 2 + 1] = (float)(i + 1) / (float)(x + 1);
	}
}

void Interp(unsigned char *src, int hg, int wd, char *wf, float *w, int x, int y, unsigned char *output)
{

	x = x - 1;
	y = y - 1;
	int r, c, i, j, nc, nr, size;
	size = 1;

	int nwd = wd * (x + 1);

	float temp;

	for (r = 0; r < hg; r++)
	{
		for (c = 0 + size - 1; c < wd - size; c++)
		{
			nr = r * (y + 1);
			nc = c * (x + 1);

			output[nr * nwd + nc] = src[r * wd + c];

			for (i = 0; i < x; i++)
			{
				nc = c * (x + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
					temp += w[i * (size * 2) + j] * (float)src[r * wd + c - size + j + 1];

				output[nr * nwd + nc] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}

	int ntemp;

	for (r = 0 + size - 1; r < hg - size; r++)
	{
		for (c = 0 * (x + 1); c < wd * (x + 1) + x; c++)
		{
			for (i = 0; i < y; i++)
			{
				nr = r * (y + 1) + i + 1;
				temp = 0;
				for (j = 0; j < size * 2; j++)
				{
					ntemp = (r - size + j + 1) * (y + 1);
					temp += w[i * (size * 2) + j] * (float)output[ntemp * nwd + c];
				}

				output[nr * nwd + c] = (unsigned char)((int)(temp + 0.5));
			}
		}
	}
}
