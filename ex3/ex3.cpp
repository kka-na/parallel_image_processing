#include <omp.h>
#include <iostream>

#define n 1000
int eq1()
{
    int s1 = 0;
#pragma omp parallel for

    for (int i = 1; i <= 10000; i++)
    {
        s1 += i;
    }
    return s1;
}
double eq2()
{
    double s2 = 1;
#pragma omp parallel for

    for (int i = 1; i <= 20; i++)
    {
        s2 *= i;
    }
    return s2;
}

int main()
{
#pragma omp parallel
    {
        printf("Hello PIP ! - %d/%d\n", omp_get_thread_num(), omp_get_max_threads());
    }

    int temp = 0;
    double a[n], b[n], c[n];
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = a[i];
    }
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }

#pragma omp parallel for private(temp)
    for (int i = 0; i <= n; i++)
    {
        temp = 2.0 * a[i];
        a[i] = temp;
        b[i] = c[i] / temp;
    }

    int isum = 0;
#pragma omp parallel for firstprivate(isum)
    for (int i = 1; i <= 1000; i++)
    {
        isum += i;
    }
    printf("[FirstPrivate Example] isum = %d\n", isum);
#pragma omp parallel for firstprivate(isum) lastprivate(isum)
    for (int i = 1; i <= 1000; i++)
    {
        isum += i;
    }
    printf("[LastPrivate Example] isum = %d\n", isum);
    isum = 0;
#pragma omp parallel for reduction(+ \
                                   : isum)
    for (int i = 1; i <= 100; i++)
    {
        isum += a[i];
    }
    printf("[Reduction Example] isum = %d\n", isum);
    isum = 0;
    if (800 <= n)
    {
#pragma omp parallel for
        for (int i = 1; i <= n; i++)
            isum += a[i];
    }
    else
    {
        for (int i = 1; i <= n; i++)
            isum += a[i];
    }
    printf("[Parallel Schedule Example] isum = %d\n", isum);
#pragma omp parallel for if (800 <= n)
    for (int i = 1; i <= n; i++)
        isum += a[i];
    printf("[If clause Example] isum = %d\n", isum);

    printf("Paralle region? = %d\n", omp_in_parallel());
#pragma omp parallel
    printf("Prallel region?=%d\n", omp_in_parallel());

    printf("Dynamic status = %d\n", omp_get_dynamic());
    printf("Serial : max threads = %d\n", omp_get_max_threads());
#pragma omp parallel
    printf("Parallel : max threads = %d\n", omp_get_max_threads());

    omp_set_nested(1);
    printf("nested status = %d\n", omp_get_nested());

    clock_t start, finish;
    int s1;
    double s2, dur1, dur2;

    start = clock();
    s1 = eq1();
    finish = clock();
    dur1 = (double)(finish - start) / CLOCKS_PER_SEC;
    start = clock();
    s2 = eq2();
    finish = clock();
    dur2 = (double)(finish - start) / CLOCKS_PER_SEC;

    printf("Parallel\nS1 = %d | %f sec\n", s1, dur1);
    printf("S2 = %f | %f sec\n", s2, dur2);

#pragma omp parallel sections
    {
#pragma omp section
        s1 = eq1();

#pragma omp section
        s2 = eq2();
    }
    printf("Section\nS1 = %d \n", s1);
    printf("S2 = %f \n", s2);

    start = clock();
    s1 = 0;
    for (int i = 1; i <= 10000; i++)
    {
        s1 += i;
    }
    finish = clock();
    dur1 = (double)(finish - start) / CLOCKS_PER_SEC;
    start = clock();
    s2 = 1;
    for (int i = 1; i <= 20; i++)
    {
        s2 *= i;
    }
    finish = clock();
    dur2 = (double)(finish - start) / CLOCKS_PER_SEC;

    printf("Serial\nS1 = %d | %f sec\n", s1, dur1);
    printf("S2 = %f | %f sec\n", s2, dur2);
    return 0;
}