#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define NL (%(block_size)d+2)
    
__constant unsigned int ng[3] = {NX,NY,NZ};
__constant unsigned int bg[3] = {NY*NZ,NZ,1};
__constant unsigned int bl[3] = {NL*NL,NL,1};
    



float3 Deriv(__local float3* Xl, unsigned int ldx, uchar dim, uchar order)
{  
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
       
    float3 out;
      
    if ((ii[dim] != 0) && (ii[dim] != (ng[dim]-1))) { 
        if (order == 1) out = Xl[ldx+bl[dim]]*(float3)(0.5) - Xl[ldx-bl[dim]]*(float3)(0.5);
        if (order == 2) out = Xl[ldx+bl[dim]] + Xl[ldx-bl[dim]] - (float3)(2)*Xl[ldx];              
    } else if (ii[dim] == 0) {
        if (order == 1) out = -(float3)(1.5)*Xl[ldx] + (float3)(2)*Xl[ldx+bl[dim]] - (float3)(0.5)*Xl[ldx+2*bl[dim]]; 
        if (order == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx+bl[dim]] + (float3)(4)*Xl[ldx+2*bl[dim]] - Xl[ldx+3*bl[dim]];
    } else if (ii[dim] == (ng[dim]-1)) {
        if (order == 1) out = (float3)(1.5)*Xl[ldx] - (float3)(2)*Xl[ldx-bl[dim]] + (float3)(0.5)*Xl[ldx-2*bl[dim]];
        if (order == 2) out = (float3)(2)*Xl[ldx] - (float3)(5)*Xl[ldx-bl[dim]] + (float3)(4)*Xl[ldx-2*bl[dim]] - Xl[ldx-3*bl[dim]];
    }
    return out;
}


__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Jacobian(__global float3* X, __global float3* J)
{
    __local float3 Xl[NL*NL*NL];
    
    unsigned int ii[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ii[0]*bg[0]+ii[1]*bg[1]+ii[2]*bg[2];
    
    unsigned int ll[3] = {get_local_id(0)+1,get_local_id(1)+1,get_local_id(2)+1};
    unsigned int ldx = ll[0]*bl[0]+ll[1]*bl[1]+ll[2]*bl[2];
    
    uchar i;
    
    Xl[ldx] = X[idx];
    
    for (i = 0; i < 3; i++) {
        if ((ll[i] == 1) && (ii[i] != 0)) Xl[ldx-bl[i]] = X[idx-bg[i]];
        if ((ll[i] == NL-2) && (ii[i] != ng[i]-1)) Xl[ldx+bl[i]] = X[idx+bg[i]];
        barrier(CLK_LOCAL_MEM_FENCE);
        J[idx + i*NX*NY*NZ] = Deriv(Xl, ldx, i, 1);
    }
}