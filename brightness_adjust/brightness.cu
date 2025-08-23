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

  int idx = (y * w + x) * ch;

  for(int i = 0; i < ch; i++){
    int v = img[idx + i] + val;
    img[idx + i] = v;
  }
}

int main(){
  int w, h, ch;
  unsigned char* h_img = stbi_load("../images/minion.png", &w, &h, &ch, 0);

  if(!h_img){
    printf("stbi error loading image");
    return 1;
  }

  size_t bytes = static_cast<size_t>(w * h) * ch;
  unsigned char* d_image{};
  cudaMalloc(&d_image, bytes);
  cudaMemcpy(d_image, h_img, bytes, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  //grid size: how many blocks to cover image
  dim3 grid((block.x + w) / block.x, (block.y + h) / block.y);
  brighten<<<grid, block>>>(d_image, w, h, ch, 50);
  cudaDeviceSynchronize();

  //load image back to host
  cudaMemcpy(h_img, d_image, bytes, cudaMemcpyDeviceToHost);
  stbi_write_png("../images/output1.png", w, h, ch, h_img, w * ch);
  stbi_image_free(h_img);

  printf("saved image!");
  return 0;
}