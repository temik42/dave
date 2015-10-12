#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define WX  %(wx)d
#define WY  %(wy)d
#define WZ  %(wz)d
#define NITER %(niter)d
    

__kernel __attribute__((reqd_work_group_size(NX,1,1)))
void box_sm_x(__global float* X, __global float* X1)
{       
    __local float Xl[NX];
    __local float Yl[NX];
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    
    int idx = iz + iy*NZ + ix*NZ*NY;
    
    Xl[ix] = X[idx];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int i,j;
    

    for (i=0; i<NITER; i++) {
        Yl[ix] = 0.f;
        for (j=-WX; j<=WX; j++) if ((ix+j >= 0) && (ix+j <= NX-1)) Yl[ix] += Xl[ix+j]/(float)(min(NX-1-ix,WX)+min(ix,WX)+1);
        barrier(CLK_LOCAL_MEM_FENCE);
        Xl[ix] = Yl[ix];
    }
    
    X1[idx] = Xl[ix]; 
}

__kernel __attribute__((reqd_work_group_size(1,NY,1)))
void box_sm_y(__global float* X, __global float* X1)
{       
    __local float Xl[NY];
    __local float Yl[NY];
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    
    int idx = iz + iy*NZ + ix*NZ*NY;
    
    Xl[iy] = X[idx];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int i,j;
    

    for (i=0; i<NITER; i++) {
        Yl[iy] = 0.f;
        for (j=-WY; j<=WY; j++) if ((iy+j >= 0) && (iy+j <= NY-1)) Yl[iy] += Xl[iy+j]/(float)(min(NY-1-iy,WY)+min(iy,WY)+1);
        barrier(CLK_LOCAL_MEM_FENCE);
        Xl[iy] = Yl[iy];
    }
    
    X1[idx] = Xl[iy]; 
}



__kernel __attribute__((reqd_work_group_size(1,1,NZ)))
void box_sm_z(__global float* X, __global float* X1)
{       
    __local float Xl[NZ];
    __local float Yl[NZ];
    
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int iz = get_global_id(2);
    
    int idx = iz + iy*NZ + ix*NZ*NY;
    
    Xl[iz] = X[idx];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int i,j;
    

    for (i=0; i<NITER; i++) {
        Yl[iz] = 0.f;
        for (j=-WZ; j<=WZ; j++) if ((iz+j >= 0) && (iz+j <= NZ-1)) Yl[iz] += Xl[iz+j]/(float)(min(NZ-1-iz,WZ)+min(iz,WZ)+1);
        barrier(CLK_LOCAL_MEM_FENCE);
        Xl[iz] = Yl[iz];
    }
    
    X1[idx] = Xl[iz]; 
}
