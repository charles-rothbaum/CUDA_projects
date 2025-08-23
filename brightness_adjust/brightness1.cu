#include <cstdio>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

__global__ void brighten(unsigned char* img, int w, int h, int ch, int val){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(x > w || y > h){
    return;
  }

  int idx = ((y * w) + x) * ch;

  for(int c = 0; c < ch; c++){
    int v = img[idx + c] + val;
    img[idx + c] = v;
  }
}

int main(){
  int w, h, ch;

  unsigned char* h_img = stbi_load("../images/minion.png", &w, &h, &ch, 0);
  if(!h_img){
    printf("error: loading image");
    return 1;
  }
  size_t bytes = static_cast<size_t>(w * h * ch);

  unsigned char* d_img{};
  cudaMalloc(&d_img, bytes);
  cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice);

  dim3 block(16,16);
  dim3 grid((w + block.x)/block.x, (h + block.y)/block.y);

  brighten<<<grid, block>>>(d_img, w, h, ch, 20);

  cudaDeviceSynchronize();

  cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_img);
  stbi_write_png("../images/output2.png", w, h, ch, h_img, w * ch);
  stbi_image_free(h_img);
  printf("saved!");
  return 0;
}