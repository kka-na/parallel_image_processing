
#include <iostream>
#include <omp.h>
#include <ctime>

using namespace std;

#define n 100

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
    cout << "\nParalle Image Processing Programming Ex3\n22212231 Kana Kim\n";
    int question_num = 0;
    cout << "\n\n፨ Enter Question number to To check the answer to the problem. ፨" << endl;
    cout << "[ Question #1 ]\nUsing parallel for to improve following Equation .\n";
    cout << "[ Question #2 ]\nUsin parallel sections to execute both Equation at same time.\n";
    cout << "[ Question #3 ]\nCompare the processing between serial and parallel.\n";

    omp_set_num_threads(omp_get_max_threads());

    while (true)
    {
        cout << "\n፨ Quit to Enter 0 ፨" << endl;
        cout << "▶▷▶ ";
        cin >> question_num;

        clock_t start, finish;

        if (question_num == 1)
        {
            start = clock();
            int s1 = eq1();
            finish = clock();
            double dur1 = (double)(finish - start) / CLOCKS_PER_SEC;
            start = clock();
            double s2 = eq2();
            finish = clock();
            double dur2 = (double)(finish - start) / CLOCKS_PER_SEC;

            cout << s1 << " " << dur1 << "sec" << endl;
            cout << s2 << " " << dur2 << "sec" << endl;
        }
        else if (question_num == 2)
        {
            int s1;
            double s2, dur1, dur2;
#pragma omp parallel sections
            {
#pragma omp section
                s1 = eq1();

#pragma omp section
                s2 = eq2();
            }
            cout << s1 << " " << dur1 << "sec" << endl;
            cout << s2 << " " << dur2 << "sec" << endl;
        }
        else if (question_num == 3)
        {
        }
        else if (question_num == 0)
        {
            cout << "End Excercise 3 ... " << endl
                 << endl;
        }
        else
        {
            cout << "Enter Number again" << endl;
        }
    }

    return 0;
}