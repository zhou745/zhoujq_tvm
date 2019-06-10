#include<cuda.h>

__device__ void load_main_D(half *D,half *shmem,int W,int C,
                            int offset_sh,int thread_per_row,int chunk_per_thread,
                            int num_per_row,int row_num_sh,int dilate){
    //calculate offset for shmem and D
    int col_id = threadIdx.x/thread_per_row;
    int row_id = (threadIdx.x/2)*chunk_per_thread%num_per_row;
    int lan_id = threadIdx.x%2;
    int offset_shmem=offset_sh+col_id*row_num_sh+row_id*16+lan_id*8;
    int offset_D=col_id*dilate*W*C+row_id*dilate*C+lan_id*8;
    
    //find the position of current thread on shmem
    half* dst=shmem+offset_shmem;
    //move D to the continuous position
    half* src=D+offset_D;
    //move the data
    for(int id=0;id<chunk_per_thread;id++){
        *(int4*)(dst+id*16) = *(int4 *)(src+id*dilate*C);
    }
}

__device__ void load_point(half *D,half *shmem,int offset_D,int offset_sh,bool fill_zero,
                            int W,int C,int num_row,int memory_h,int memory_w,int memory_c,int dilate){
    //this function is used to load data of the corner point
    //note that only the first two thread in a warp should call this function
    half *dst = shmem+offset_sh;
    if(fill_zero){
        *(int4 *)(dst+num_row*memory_h+16*memory_w+8*memory_c) = make_int4(0,0,0,0);
    } else {
        half *src = D+offset_D;
        *(int4 *)(dst+num_row*memory_h+16*memory_w+8*memory_c) = *(int4 *)(src+memory_h*W*C*dilate+memory_w*C*dilate+8*memory_c);
    }
}

__device__ void load_matrix_D_0(half *D,half *shmem,int N,int H,int W,int C,int dc,int dilate){
    //calculate the initial position of data to be loaded
    int posi =64*blockIdx.x%(N*H*W);
    int dN = posi/(H*W);
    posi = posi%(H*W)/64;
    //calculate the id of current block
    int h_id = posi/(W/8);
    int w_id = posi%(W/8);
    //calculate the index of first element 
    int dh = (h_id/dilate)*8*dilate+h_id%dilate;
    int dw = (w_id/dilate)*8*dilate+w_id%dilate;
    // move D to current warp
    bool fill_zero = true;
    D+=(dN*H*W*C+dh*W*C+dw*C+dc);
    
    //load the current warp based on the position of the warp
    load_main_D(D,shmem,W,C,176,16,1,8,160,dilate);

    //fill the outside boarder
    if(threadIdx.x<2){
        //load top left point
        fill_zero = dh<dilate||dw<dilate;
        load_point(D,shmem,-(W+dilate)*C,0,fill_zero,
                    W,C,160,0,0,threadIdx.x,dilate);
    } else if(threadIdx.x<4){
        //load top right point
        fill_zero = dh<dilate||dw>=(W-dilate*8);
        load_point(D,shmem,-(W-8*dilate)*C,144,fill_zero,
                   W,C,160,0,0,threadIdx.x-2,dilate);
    } else if(threadIdx.x<6) {
        //load bot left point
        fill_zero = dh>=(H-dilate*8)||dw<dilate;
        load_point(D,shmem,(8*W-dilate)*C,1440,fill_zero,
                    W,C,160,0,0,threadIdx.x-4,dilate);
    } else if(threadIdx.x<8){
        //load bot right point
        fill_zero = dh>=(H-dilate*8)||dw>=(W-dilate*8);
        load_point(D,shmem,(W+dilate)*C*8,1584,fill_zero,
                    W,C,160,0,0,threadIdx.x-6,dilate);
    } else if(threadIdx.x<24){
        //load the top edge
        fill_zero = dh<dilate;
        load_point(D,shmem,-W*C,16,fill_zero,
        W,C,160,0,(threadIdx.x-8)/2,(threadIdx.x-8)%2,dilate);
    } else if(threadIdx.x<40){
        //load the bot edge
        fill_zero = dh>=(H-dilate*8);
        load_point(D,shmem,8*W*C,1456,fill_zero,
            W,C,160,0,(threadIdx.x-24)/2,(threadIdx.x-24)%2,dilate);
    } else if(threadIdx.x<56){
        //load the left edge
        fill_zero = dw<dilate;
        load_point(D,shmem,-C*dilate,160,fill_zero,
        W,C,160,(threadIdx.x-40)/2,0,(threadIdx.x-40)%2,dilate);
    } else if(threadIdx.x<72){
        //load the right edge
        fill_zero = dw>=(W-dilate*8);
        load_point(D,shmem,8*dilate*C,304,fill_zero,
            W,C,160,(threadIdx.x-56)/2,0,(threadIdx.x-56)%2,dilate);
    }
}

__device__ void load_matrix_D_1(half *D,half *shmem,int N,int H,int W,int C,int dc,int dilate){
    //calculate the initial position of data to be loaded
    int posi = 256*blockIdx.x%(N*H*W);
    int dN = posi/(H*W);
    posi = posi%(H*W)/256;
    //calculate the id of current block
    int h_id = posi/(W/16);
    int w_id = posi%(W/16);
    //calculate the index of first element 
    int dh = (h_id/dilate)*16*dilate+h_id%dilate;
    int dw = (w_id/dilate)*16*dilate+w_id%dilate;

    // move D to current warp
    bool fill_zero = true;
    D+=(dN*H*W*C+dh*W*C+dw*C+dc);

    //load the current warp based on the position of the warp
    load_main_D(D,shmem,W,C,304,8,4,16,288,dilate);

    //fill the outside boarder
    if(threadIdx.x<2){
        //load top left point
        fill_zero = dh<dilate||dw<dilate;
        load_point(D,shmem,-(W+dilate)*C,0,fill_zero,
                    W,C,288,0,0,threadIdx.x,dilate);
    } else if(threadIdx.x<4){
        //load top right point
        fill_zero = dh<dilate||dw>=(W-dilate*16);
        load_point(D,shmem,-(W-16*dilate)*C,272,fill_zero,
                   W,C,288,0,0,threadIdx.x-2,dilate);
    } else if(threadIdx.x<6) {
        //load bot left point
        fill_zero = dh>=(H-dilate*16)||dw<dilate;
        load_point(D,shmem,(16*W-dilate)*C,4896,fill_zero,
                    W,C,288,0,0,threadIdx.x-4,dilate);
    } else if(threadIdx.x<8){
        //load bot right point
        fill_zero = dh>=(H-dilate*16)||dw>=(W-dilate*16);
        load_point(D,shmem,(W+dilate)*C*16,5168,fill_zero,
                    W,C,288,0,0,threadIdx.x-6,dilate);
    }
    if(threadIdx.x<32){
        //load the top edge
        fill_zero = dh<dilate;
        load_point(D,shmem,-W*C,16,fill_zero,
        W,C,288,0,(threadIdx.x)/2,(threadIdx.x)%2,dilate);
    } else if(threadIdx.x<64){
        //load the bot edge
        fill_zero = dh>=(H-dilate*16);
        load_point(D,shmem,16*W*C,4912,fill_zero,
            W,C,288,0,(threadIdx.x-32)/2,(threadIdx.x-32)%2,dilate);
    } else if(threadIdx.x<96){
        //load the left edge
        fill_zero = dw<dilate;
        load_point(D,shmem,-C*dilate,288,fill_zero,
        W,C,288,(threadIdx.x-64)/2,0,(threadIdx.x-64)%2,dilate);
    } else{
        //load the right edge
        fill_zero = dw>=(W-dilate*16);
        load_point(D,shmem,16*dilate*C,560,fill_zero,
            W,C,288,(threadIdx.x-96)/2,0,(threadIdx.x-96)%2,dilate);
    }
}

__device__ void load_matrix_F_1(half *F,half *shmem,int offset_F,int K,int C,\
                             int N,int H,int W,int dc,int dRS){
}

__device__ void load_matrix_F_0(half *F,half *shmem,int offset_F,int K,int C,\
                             int N,int H,int W,int dc,int dRS){
    //move shmem to the loading position of current thread
    half *dst=shmem+offset_F;

    //find the line of current location
    int posi=blockIdx.x*64;
    int dK=posi/(N*H*W)*64+threadIdx.x/2;
   
    //set the F pointer to loading location
    half *src=F+dK*9*C+dRS*C+dc;
    //fill the shared memory
    if(dK>=K){//out of range for K
        for(int id=0;id<8;id++){
            dst[id]=static_cast<half>(0.);
        }
    } else{
        *(int4 *)dst=*(int4*)src;
    }
}

__device__ void im2col_0(half *shmem,int offset_D,int ker_id){
    //find the position of current thread
    int colid = threadIdx.x/16;
    int rowid = (threadIdx.x/2)%8;
    int depth = threadIdx.x%2;
    half *src=shmem+ker_id/3*160+ker_id%3*16+colid*160+rowid*16+depth*8;
    
    half *dst=shmem+offset_D;

    *(int4*)dst=*(int4*)src;
}

__device__ void load_matrix_D(half * D,half * shmem,int N,int H,int W,int C,int dc,int dilate,int version){
    //calculate the initial position of data to be loaded 
    if(version==1){
        load_matrix_D_1(D,shmem,N,H,W,C,dc,dilate);
    } else {
        load_matrix_D_0(D,shmem,N,H,W,C,dc,dilate);
    }  
}

__device__ void load_matrix_F(half *F,half *shmem,int offset_F,int K,int C,
                             int N,int H,int W,int dc,int dRS,
                             int version){
    if(version==1) {
        load_matrix_F_1(F,shmem,offset_F,K,C,N,H,W,dc,dRS);
    } else {
        load_matrix_F_0(F,shmem,offset_F,K,C,N,H,W,dc,dRS);
    }
}

__device__ void im2col(half *shmem,int offset_D,int ker_id,int version){
    if(version==1){
        //im2col_128(shmem,offset_D,ker_id);
    } else {
        im2col_0(shmem,offset_D,ker_id);
    }
}


__device__ void debug_copy(half *src,half *dst,int offset, int num){
    for(int id=0;id<num;id++){
        dst[id]=src[offset+id];
    }
}