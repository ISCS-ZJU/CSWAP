/* minimal code example showing how to call the zfp (de)compressor */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp/zfp.h"
#include "zfp/bitstream.c"
#include "zfp/cuZFP.h"
#include <sys/time.h>

/* compress or decompress array */
static int
compress(float* array, size_t nx, size_t ny, size_t nz, double tolerance, zfp_bool decompress,float* ori)
{
  int status = 0;    /* return value: 0 = success */
  zfp_type type;     /* array scalar type */
  zfp_field* field;  /* array meta data */
  zfp_stream* zfp;   /* compressed stream */
  void* buffer;      /* storage for compressed stream */
  size_t bufsize;    /* byte size of compressed buffer */
  bitstream* stream; /* bit stream to write to or read from */
  size_t zfpsize;    /* byte size of compressed stream */

  /* allocate meta data for the 3D array a[nz][ny][nx] */
  type = zfp_type_float;
  field = zfp_field_3d(array, type, nx, ny, nz);

  /* allocate meta data for a compressed stream */
  zfp = zfp_stream_open(NULL);

  /* set compression mode and parameters via one of four functions */
/*  zfp_stream_set_reversible(zfp); */
/*  zfp_stream_set_rate(zfp, rate, type, zfp_field_dimensionality(field), zfp_false); */
/*  zfp_stream_set_precision(zfp, precision); */
  zfp_stream_set_accuracy(zfp, tolerance);

  /* allocate buffer for compressed data */
  bufsize = zfp_stream_maximum_size(zfp, field);
  buffer = malloc(bufsize);
  printf("bufsize: %zu MB\n",bufsize/1024/1024);
  /* associate bit stream with allocated buffer */
  stream = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  /* compress or decompress entire array */
  if (decompress) {
    /* read compressed stream and decompress and output array */
    zfpsize = fread(buffer, 1, bufsize, stdin);
    cuda_decompress(zfp, field);
    /**
    if (!zfp_decompress(zfp, field)) {
      fprintf(stderr, "decompression failed\n");
      status = EXIT_FAILURE;
    }
    else
      fwrite(array, sizeof(double), zfp_field_size(field, NULL), stdout);
      **/
  }
  else {
    /* compress array and output compressed stream */
    zfpsize = cuda_compress(zfp, field);
    if (!zfpsize) {
      fprintf(stderr, "compression failed\n");
      status = EXIT_FAILURE;
    }
    else{
      //fwrite(buffer, 1, zfpsize, stdout);
      printf("zfpsize : %lf MB\n",(double)zfpsize/1024.0/1024.0);
    }
    //array[0] = 2.3;
    memset(array,0,nx * ny * nz * sizeof(float));
    cuda_decompress(zfp, field);
    
    //FILE* f = fopen("com.txt","w");
    //fwrite(array, sizeof(double), zfp_field_size(field, NULL), stdout);
    int cnt = 0;
    for (int k = 0; k < nz; k++){
      for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++) {
          //array[i + nx * (j + ny * k)] = exp(-(x * x + y * y + z * z));
          //printf("%lf",array[i + nx * (j + ny * k)]);
          if(array[i + nx * (j + ny * k)]!=ori[i + nx * (j + ny * k)]){
              //printf("%lf:%lf\n",array[i + nx * (j + ny * k)],ori[i + nx * (j + ny * k)]);
              cnt++;
          }

          //fprintf(f,"%lf ",array[i + nx * (j + ny * k)]);
        }
      }
    }
    printf("error cnt:%d\n",cnt);
    //fclose(f);**/
  }

  /* clean up */
  // zfp_field_free(field);
  // zfp_stream_close(zfp);
  // stream_close(stream);
  // free(buffer);
  // free(array);

  return status;
}

int main(int argc, char* argv[])
{
  /* use -d to decompress rather than compress data */
  zfp_bool decompress = (argc == 2 && !strcmp(argv[1], "-d"));

  /* allocate 100x100x100 array of doubles */
  size_t nx = 2560;
  size_t ny = 512;
  size_t nz = 50;
  printf("All tensor size =  %zu MB\n", 4*nx*ny*nz/ 1024 / 1024);
  float* array = malloc(nx * ny * nz * sizeof(double));
  //FILE* f = fopen("ori.txt","w");
  if (!decompress) {
    /* initialize array to be compressed */
    int i, j, k;
    for (k = 0; k < nz; k++)
      for (j = 0; j < ny; j++)
        for (i = 0; i < nx; i++) {
          double x = 2.0 * i / nx;
          double y = 2.0 * j / ny;
          double z = 2.0 * k / nz;
          // array[i + nx * (j + ny * k)] = exp(-(x * x + y * y + z * z));
          float a =  (float)((rand() % 20)+x+y+z) / (x+y+z+rand() % 3)* pow(-1,k+j+i);
          array[i + nx * (j + ny * k)] = a;
          // printf("%f ",a);
          //fprintf(f,"%lf ",array[i + nx * (j + ny * k)]);
        }
  }
  //fclose(f);

  float* ori = malloc(nx * ny * nz * sizeof(float));
  memcpy(ori,array,nx * ny * nz * sizeof(float));
  
  float array1[1000];
  
  for(int i=0;i<1000;i++)
    array1[i] = array[i];
    //printf("Begin: %f ",array[i]);
  //printf("\n");
  /* compress or decompress array */
  compress(array, nx, ny, nz, 1e-6, decompress, ori);
  float error_totle = 0;
  for(int i=0;i<1000;i++)
  {
    float error = (array1[i] - array[i]) / array1[i];
    if(error<0)
      error = error * (-1.0);
    error_totle += error;
    // printf("Begin: %f ",(array1[i] - array[i]));
  }
  printf("Final accuray loss is -> %f\n",error_totle/1000.0);
  return 0;
}
