#include <cstdio>
#include <cuda_runtime.h>

//here we are making single headers for the single-header libraries 
//this means that the headers are created here, once, and in any other file just
//use #include "stb_image.h"
//in other words the functions of that library are part of this programs binary now.
//can also pass the definitions on the command for nvcc compiling. hello
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

__global__ void brighten(unsigned char* img, int w, int h, int ch, int value){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h) return;

    int idx = (y * w + x) * ch;

    for(int c = 0; c < ch; c++){
        int v = img[idx + c] + value;
        img[idx + c] = v;
    }
}

int main(){
    int w, h, ch;
    unsigned char* h_img = stbi_load("input.png", &w, &h, &ch, 0); //host image. pass in dereferenced integers (pointers) so the function can modify them in place.
    if (!h_img){
        printf("stbi load failed");
        return 1;
    }

    //size_t is the data type returned by sizeof().
    //on a 64 bit machine, it is a 64 bit unsigned int
    //safe for holding the sizes of large arrays.

    //static cast is the safe type conversion operator. static_cast<new_type>(var)
    size_t bytes = static_cast<size_t>(w) * h * ch;
    unsigned char* d_img{}; // {} same as assigning nullptr. this is a pointer that points to nothing.
    cudaMalloc(&d_img, bytes); //reserve bytes of data on the GPU global memory, and give back a pointer to that memory
    //in cudaMalloc i pass a pointer to the pointer d_img because otherwise i would be giving it 
    //the actual d_img, which is null at this point. 

    cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice); //move the host image into device image.
    
    dim3 block(16, 16); //16 by 16 threads per block. dim3 is just the CUDA struct with xyz components.
    dim3 grid((w + block.x - 1) / block.x, (h + block.y -1) / block.y);
    brighten<<<grid, block>>>(d_img, w, h, ch, 50);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_img);

    stbi_write_png("output.png", w, h, ch, h_img, w*ch);
    stbi_image_free(h_img);

    printf("saved output.png!\n");
    return 0;

}