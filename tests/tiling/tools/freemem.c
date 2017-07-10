#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char*argv [])
{

  long n = 1024l*1024l*1024l*30l;
  time_t t;

  srand((unsigned) time(&t));

  int* a = (int*)calloc(n, sizeof(int));

  int b[100];
  for (int i = 0; i < 100; i++) {
    int r = rand() % n;
    b[i] = r;
    a[r] = rand() % n;
  }

  int sum = 0;
  for (int i = 0; i < 100; i++) {
    sum += a[b[i]];
  }

  printf("Sample: %d): freed DRAM.\n", sum);
  fflush(stdout);

  free(a);

  return 0;
}
