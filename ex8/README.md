# INHA Univ. Parallel Image Processing Programming for Graduated Students EX8

## ./ex8

```
kana@Alienware:~/Documents/Class/Parallel image processing programming/ex8/build$ cmake .. && make && ./ex8
-- Configuring done
-- Generating done
-- Build files have been written to: /home/kana/Documents/Class/Parallel image processing programming/ex8/build
Consolidate compiler generated dependencies of target ex8
[ 33%] Building CXX object CMakeFiles/ex8.dir/ex8.cpp.o
/home/kana/Documents/Class/Parallel image processing programming/ex8/ex8.cpp: In function ‘int main()’:
/home/kana/Documents/Class/Parallel image processing programming/ex8/ex8.cpp:84:55: warning: unknown conversion type character ‘\x0a’ in format [-Wformat=]
   84 |         printf("%s more Faster than Global about %d % \n", types[i].c_str(), int((ptime[0] / ptime[i]) * 100));
      |                                                       ^~
[ 66%] Linking CXX executable ex8
[100%] Built target ex8
Global Memory Processing Time(8bit): 0.105812 sec
Shared Memory Processing Time(8bit): 0.019228 sec
Constant Memory Processing Time(8bit): 0.018523 sec
Shared more Faster than Global about 550 %
Constant more Faster than Global about 571 %
```

<p align="center">
  <p>Global</p>
  <img src="./result/Global.jpg"/>
  <p>Shared</p>
  <img src="./result/Shared.jpg"/>
  <p>Constant</p>
  <img src="./result/Constant.jpg"/>
</p>
