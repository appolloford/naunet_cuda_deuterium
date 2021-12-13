#include <math.h>
/* */
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

// clang-format off
__device__ int EvalRates(realtype *k, realtype *y, NaunetData *u_data) {

    realtype nH = u_data->nH;
    realtype Tgas = u_data->Tgas;
    realtype user_crflux = u_data->user_crflux;
    realtype user_Av = u_data->user_Av;
    realtype user_GtoDN = u_data->user_GtoDN;
    
    
    // clang-format on

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    if (Tgas>5.0 && Tgas<20.0) { k[0] = 8.160e-10*exp(-1.649e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[1] = 5.880e-10*exp(-1.982e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[2] = 2.980e-10*exp(+6.900e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[3] = 3.460e-10*exp(+6.900e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[4] = 8.030e-10*exp(-3.260e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[5] = 1.500e-09*exp(-1.362e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[6] = 8.840e-09*exp(-1.700e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[7] = 1.040e-10;  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[8] = 4.000e-10*exp(+1.900e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[9] = 9.670e-11*exp(+1.400e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[10] = 5.710e-11*exp(-3.220e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[11] = 3.110e-10*exp(+7.100e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[12] = 4.930e-10*exp(-9.500e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[13] = 6.080e-10*exp(+1.080e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[14] = 5.710e-10*exp(-2.580e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[15] = 2.870e-11*exp(+3.800e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[16] = 1.700e-10*exp(+4.400e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[17] = 2.220e-10*exp(+4.700e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[18] = 1.110e-09*exp(-3.500e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[19] = 2.460e-10*exp(-2.265e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[20] = 1.020e-09*exp(-2.561e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[21] = 1.480e-10*exp(-5.880e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[22] = 9.320e-09*exp(-9.460e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[23] = 1.260e-09*exp(-6.000e-02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[24] = 6.040e-10*exp(-8.880e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[25] = 1.310e-10*exp(-1.404e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[26] = 5.580e-10*exp(-8.270e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[27] = 6.540e-10*exp(-1.740e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[28] = 4.670e-11*exp(+8.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[29] = 1.640e-10*exp(-6.310e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[30] = 8.310e-11*exp(+9.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[31] = 1.680e-10*exp(+7.700e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[32] = 2.190e-10*exp(+7.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[33] = 3.500e-09*exp(+4.100e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[34] = 5.080e-09*exp(+8.000e-02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[35] = 3.020e-10*exp(+1.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[36] = 4.080e-10*exp(-6.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[37] = 3.060e-10*exp(-5.900e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[38] = 2.420e-10*exp(-8.000e-02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[39] = 4.810e-10*exp(+4.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[40] = 5.390e-10*exp(-6.000e-02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[41] = 8.020e-10*exp(-9.000e-02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[42] = 7.500e-10*exp(+1.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[43] = 5.590e-10*exp(-2.490e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[44] = 1.030e-09*exp(+8.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[45] = 7.830e-12*exp(-2.378e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[46] = 9.480e-12*exp(-1.466e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[47] = 2.840e-10*exp(-8.850e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[48] = 4.120e-10*exp(+5.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[49] = 1.890e-10*exp(-3.310e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[50] = 7.320e-10*exp(-3.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[51] = 1.930e-10*exp(+6.400e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[52] = 2.520e-12*exp(-1.501e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[53] = 3.880e-12*exp(-6.510e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[54] = 6.800e-12*exp(-1.817e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[55] = 1.030e-10*exp(-9.680e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[56] = 8.640e-11*exp(+3.800e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[57] = 6.410e-11*exp(-2.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[58] = 3.020e-10*exp(+6.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[59] = 1.490e-10*exp(+9.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[60] = 5.240e-10*exp(+5.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[61] = 2.020e-10*exp(-3.550e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[62] = 3.260e-10*exp(-1.373e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[63] = 4.490e-10*exp(-2.314e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[64] = 7.090e-10*exp(-1.688e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[65] = 2.650e-11*exp(-2.339e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[66] = 7.330e-11*exp(-1.580e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[67] = 5.940e-10*exp(-5.460e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[68] = 2.840e-10*exp(+5.800e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[69] = 1.560e-11*exp(-3.252e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[70] = 3.480e-10*exp(-1.936e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[71] = 4.610e-10*exp(-2.817e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[72] = 4.160e-10*exp(-1.711e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[73] = 8.350e-12*exp(-1.711e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[74] = 1.650e-11*exp(-1.946e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[75] = 8.150e-11*exp(-1.560e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[76] = 6.820e-10*exp(-1.034e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[77] = 4.170e-10*exp(+3.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[78] = 4.820e-11*exp(+1.010e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[79] = 6.780e-10*exp(+2.300e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[80] = 5.410e-10*exp(-8.500e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[81] = 1.410e-10*exp(+1.050e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[82] = 2.070e-11*exp(-8.630e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[83] = 2.570e-10*exp(+5.500e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[84] = 7.490e-10*exp(-6.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[85] = 2.270e-10*exp(+8.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[86] = 1.600e-10*exp(-1.100e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[87] = 3.930e-11*exp(-2.100e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[88] = 6.640e-10*exp(-2.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[89] = 5.390e-10*exp(+4.400e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[90] = 1.310e-10*exp(-1.800e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[91] = 3.950e-11*exp(-8.850e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[92] = 2.740e-10*exp(+3.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[93] = 8.750e-10*exp(-5.300e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[94] = 1.630e-10*exp(+1.570e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[95] = 8.010e-11*exp(+9.400e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[96] = 1.540e-11*exp(-1.455e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[97] = 1.170e-11*exp(-5.700e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[98] = 9.470e-11*exp(-2.373e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[99] = 4.680e-11*exp(-1.462e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[100] = 3.360e-10*exp(-1.800e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[101] = 1.090e-10*exp(+7.800e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[102] = 3.700e-10*exp(-5.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[103] = 7.830e-12*exp(-2.022e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[104] = 2.120e-11*exp(-1.076e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[105] = 3.590e-11*exp(-2.851e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[106] = 7.790e-11*exp(-1.967e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[107] = 2.900e-10*exp(-4.830e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[108] = 1.360e-10*exp(+1.500e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[109] = 2.070e-10*exp(+1.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[110] = 1.100e-10*exp(+2.700e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[111] = 2.840e-10*exp(+3.800e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[112] = 1.650e-10*exp(-3.449e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[113] = 9.570e-10*exp(-2.393e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[114] = 1.900e-10*exp(-2.627e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[115] = 1.530e-09*exp(-6.560e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[116] = 1.070e-10*exp(-3.939e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[117] = 7.850e-11*exp(-2.969e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[118] = 9.430e-10*exp(-2.374e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[119] = 6.890e-10*exp(-1.897e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[120] = 1.530e-10*exp(-3.038e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[121] = 9.600e-11*exp(-2.136e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[122] = 9.060e-10*exp(-6.620e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[123] = 7.700e-10*exp(-1.700e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[124] = 5.160e-11*exp(-1.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[125] = 4.360e-11*exp(-3.110e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[126] = 1.420e-10*exp(-1.310e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[127] = 8.230e-10*exp(+1.300e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[128] = 3.740e-11*exp(-8.590e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[129] = 1.050e-10*exp(-3.560e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[130] = 9.120e-11*exp(-3.650e+00*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[131] = 2.930e-10*exp(-1.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[132] = 7.590e-10*exp(+5.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[133] = 3.640e-11*exp(-5.000e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[134] = 1.920e-10*exp(+7.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[135] = 1.110e-10*exp(+5.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[136] = 2.770e-10*exp(+7.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[137] = 6.520e-10*exp(-9.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[138] = 5.750e-11*exp(-1.377e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[139] = 7.310e-11*exp(-5.030e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[140] = 4.280e-11*exp(-8.550e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[141] = 5.820e-10*exp(+8.000e-02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[142] = 5.810e-10*exp(-4.000e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[143] = 1.080e-10*exp(-2.067e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[144] = 8.740e-11*exp(-2.513e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[145] = 2.650e-10*exp(-1.543e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[146] = 5.970e-10*exp(-4.630e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[147] = 1.400e-10*exp(-2.474e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[148] = 1.630e-10*exp(-1.605e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[149] = 1.080e-10*exp(-1.984e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[150] = 1.200e-10*exp(-1.052e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[151] = 2.460e-10*exp(+2.300e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[152] = 1.480e-10*exp(+4.900e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[153] = 1.110e-10*exp(-4.660e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[154] = 6.120e-10*exp(-4.500e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[155] = 2.140e-10*exp(-8.470e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[156] = 1.630e-10*exp(-1.305e+02*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[157] = 6.660e-10*exp(-4.580e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[158] = 4.470e-11*exp(+2.600e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[159] = 5.640e-11*exp(-7.200e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[160] = 5.870e-10*exp(+1.900e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[161] = 3.210e-10*exp(-3.830e+01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[162] = 2.780e-10*exp(+4.700e-01*invT);  }
        
    if (Tgas>5.0 && Tgas<20.0) { k[163] = 3.240e-10*exp(-8.520e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[164] = 1.238e-17*pow((T32),
        (+5.000e-01))/y[IDX_HI]*nH;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[165] = 3.713e-17*pow((T32),
        (+5.000e-01))/y[IDX_HI]*nH;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[166] = 2.475e-17*pow((T32),
        (+5.000e-01))/(0.5*y[IDX_HI] + 0.5*y[IDX_DI])*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[167] = 1.650e-22*pow((T32),
        (+5.000e-01))/y[IDX_DI]*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[168] = 3.300e-22*pow((T32),
        (+5.000e-01))/y[IDX_DI]*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[169] = 7.428e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[170] = 5.083e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[171] = 1.948e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[172] = 1.017e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[173] = 1.694e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[174] = 8.472e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[175] = 1.694e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[176] = 8.472e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[177] = 2.541e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[178] = 4.914e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[179] = 4.914e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[180] = 1.948e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[181] = 1.017e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[182] = 1.948e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[183] = 4.236e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[184] = 8.472e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[185] = 1.271e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[186] = 1.271e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[187] = 4.236e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[188] = 8.472e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[189] = 1.101e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[190] = 3.727e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[191] = 7.541e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[192] = 3.727e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[193] = 7.541e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[194] = 1.101e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[195] = 1.355e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[196] = 6.863e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[197] = 1.017e-15*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[198] = 3.474e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[199] = 3.474e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[200] = 9.319e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[201] = 9.319e-16*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[202] = 1.020e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[203] = 4.600e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[204] = 4.600e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[205] = 5.000e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[206] = 2.100e+00*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[207] = 2.800e+00*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[208] = 1.000e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[209] = 1.000e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[210] = 1.000e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[211] = 1.000e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[212] = 1.000e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[213] = 2.200e-02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[214] = 2.200e-02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[215] = 2.200e-02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[216] = 2.200e-02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[217] = 2.200e-02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[218] = 2.200e-02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[219] = 3.000e-04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[220] = 3.000e-04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[221] = 3.000e-04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[222] = 3.000e-04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[223] = 3.000e-04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[224] = 3.000e-04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[225] = 9.300e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[226] = 9.300e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[227] = 9.300e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[228] = 9.300e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[229] = 9.300e-01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[230] = 1.760e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[231] = 1.760e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[232] = 1.120e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[233] = 1.120e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[234] = 5.000e+00*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[235] = 5.000e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[236] = 5.000e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[237] = 4.820e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[238] = 4.940e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[239] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[240] = 1.170e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[241] = 5.100e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[242] = 5.100e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[243] = 5.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[244] = 5.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[245] = 1.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[246] = 1.120e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[247] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[248] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[249] = 5.000e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[250] = 5.000e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[251] = 5.000e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[252] = 1.710e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[253] = 9.700e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[254] = 9.700e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[255] = 9.700e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[256] = 9.700e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[257] = 3.120e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[258] = 3.120e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[259] = 4.210e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[260] = 4.210e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[261] = 1.170e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[262] = 1.170e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[263] = 3.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[264] = 3.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[265] = 1.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[266] = 1.000e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[267] = 1.500e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[268] = 8.000e+01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[269] = 8.000e+01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[270] = 8.000e+01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[271] = 8.000e+01*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[272] = 6.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[273] = 6.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[274] = 6.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[275] = 1.500e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[276] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[277] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[278] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[279] = 7.500e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[280] = 1.500e+03*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[281] = 2.370e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[282] = 7.300e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[283] = 7.300e+02*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[284] = 1.060e+04*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[285] = 5.000e+00*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[286] = 3.000e+00*user_crflux;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[287] =
        1.000e-11*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[288] =
        4.600e-12*exp(-3.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[289] =
        4.600e-12*exp(-3.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[290] =
        2.600e-10*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[291] =
        2.600e-10*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[292] =
        2.600e-10*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[293] =
        2.600e-10*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[294] =
        2.600e-10*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[295] =
        2.600e-10*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[296] =
        7.200e-12*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[297] =
        7.200e-12*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[298] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[299] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[300] =
        1.700e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[301] =
        1.700e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[302] =
        1.700e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[303] =
        1.700e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[304] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[305] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[306] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[307] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[308] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[309] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[310] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[311] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[312] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[313] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[314] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[315] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[316] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[317] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[318] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[319] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[320] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[321] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[322] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[323] =
        2.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[324] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[325] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[326] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[327] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[328] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[329] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[330] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[331] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[332] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[333] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[334] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[335] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[336] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[337] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[338] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[339] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[340] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[341] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[342] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[343] =
        7.900e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[344] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[345] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[346] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[347] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[348] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[349] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[350] =
        2.400e-07*exp(-9.000e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[351] =
        2.160e-10*exp(-2.610e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[352] =
        4.700e-11*exp(-2.600e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[353] =
        1.000e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[354] =
        1.400e-10*exp(-1.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[355] =
        1.400e-10*exp(-1.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[356] =
        2.900e-10*exp(-2.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[357] =
        2.900e-10*exp(-2.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[358] =
        1.000e-09*exp(-2.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[359] =
        3.100e-11*exp(-2.540e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[360] =
        3.400e-11*exp(-2.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[361] =
        3.400e-11*exp(-2.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[362] =
        3.400e-11*exp(-2.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[363] =
        3.400e-11*exp(-2.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[364] =
        3.400e-11*exp(-2.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[365] =
        5.000e-12*exp(-3.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[366] =
        4.000e-10*exp(-1.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[367] =
        4.000e-10*exp(-1.500e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[368] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[369] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[370] =
        3.000e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[371] =
        2.000e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[372] =
        3.300e-10*exp(-1.400e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[373] =
        6.200e-12*exp(-3.100e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[374] =
        1.680e-10*exp(-1.660e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[375] =
        1.680e-10*exp(-1.660e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[376] =
        1.600e-12*exp(-3.100e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[377] =
        1.600e-12*exp(-3.100e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[378] =
        1.000e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[379] =
        1.000e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[380] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[381] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[382] =
        1.000e-10*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[383] =
        1.000e-09*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[384] =
        2.600e-10*exp(-2.280e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[385] =
        5.000e-11*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[386] =
        5.000e-11*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[387] =
        5.000e-11*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[388] =
        5.000e-11*exp(-1.700e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[389] =
        1.000e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[390] =
        1.000e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[391] =
        1.000e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[392] =
        3.130e-10*exp(-2.030e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[393] =
        3.280e-10*exp(-1.630e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[394] =
        3.280e-10*exp(-1.630e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[395] =
        3.280e-10*exp(-1.630e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[396] =
        3.280e-10*exp(-1.630e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[397] =
        2.100e-11*exp(-3.100e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[398] =
        2.100e-11*exp(-3.100e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[399] =
        2.100e-11*exp(-3.100e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[400] =
        5.480e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[401] =
        5.480e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[402] =
        5.870e-10*exp(-5.300e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[403] =
        5.870e-10*exp(-5.300e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[404] =
        2.460e-10*exp(-2.110e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[405] =
        2.460e-10*exp(-2.110e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[406] =
        5.480e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[407] =
        5.480e-10*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[408] =
        1.700e-10*exp(-5.300e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[409] =
        1.700e-10*exp(-5.300e-01*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[410] =
        2.110e-10*exp(-1.520e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[411] =
        2.110e-10*exp(-1.520e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[412] =
        2.110e-10*exp(-1.520e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[413] =
        2.110e-10*exp(-1.520e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[414] =
        1.730e-10*exp(-2.590e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[415] =
        1.730e-10*exp(-2.590e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[416] =
        1.730e-10*exp(-2.590e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[417] =
        1.290e-09*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[418] =
        1.000e-11*exp(-2.000e+00*user_Av);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[419] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[420] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[421] = 4.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[422] = 4.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[423] = 4.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[424] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[425] = 2.900e-09*pow((T32),
        (-3.330e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[426] = 2.900e-09*pow((T32),
        (-3.330e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[427] = 2.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[428] = 2.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[429] = 3.930e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[430] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[431] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[432] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[433] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[434] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[435] = 8.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[436] = 8.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[437] = 8.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[438] = 8.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[439] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[440] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[441] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[442] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[443] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[444] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[445] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[446] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[447] = 6.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[448] = 6.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[449] = 8.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[450] = 8.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[451] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[452] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[453] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[454] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[455] = 1.900e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[456] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[457] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[458] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[459] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[460] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[461] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[462] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[463] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[464] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[465] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[466] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[467] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[468] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[469] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[470] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[471] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[472] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[473] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[474] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[475] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[476] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[477] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[478] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[479] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[480] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[481] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[482] = 4.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[483] = 3.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[484] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[485] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[486] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[487] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[488] = 3.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[489] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[490] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[491] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[492] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[493] = 3.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[494] = 3.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[495] = 3.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[496] = 3.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[497] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[498] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[499] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[500] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[501] = 1.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[502] = 1.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[503] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[504] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[505] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[506] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[507] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[508] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[509] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[510] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[511] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[512] = 8.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[513] = 8.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[514] = 8.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[515] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[516] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[517] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[518] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[519] = 3.610e-13*pow((T32),
        (+2.100e+00))*exp(-3.080e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[520] = 3.610e-13*pow((T32),
        (+2.100e+00))*exp(-3.080e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[521] = 3.770e-11*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[522] = 3.770e-11*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[523] = 7.100e-12*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[524] = 7.100e-12*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[525] = 3.550e-12*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[526] = 3.550e-12*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[527] = 6.390e-11*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[528] = 6.390e-11*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[529] = 3.195e-11*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[530] = 3.195e-11*pow((T32),
        (-1.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[531] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[532] = 5.300e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[533] = 5.300e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[534] = 1.500e-11*exp(-2.000e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[535] = 4.050e-10*pow((T32),
        (-1.430e+00))*exp(-3.500e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[536] = 5.000e-12*exp(-9.000e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[537] = 1.200e-11*pow((T32),
        (-1.300e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[538] = 1.200e-11*pow((T32),
        (-1.300e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[539] = 3.800e-11*pow((T32),
        (-4.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[540] = 3.800e-11*pow((T32),
        (-4.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[541] = 1.730e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[542] = 1.730e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[543] = 1.730e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[544] = 1.730e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[545] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[546] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[547] = 1.600e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[548] = 2.400e-11*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[549] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[550] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[551] = 7.010e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[552] = 7.010e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[553] = 1.000e-11*exp(-1.000e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[554] = 1.000e-11*exp(-1.000e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[555] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[556] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[557] = 2.810e-13*exp(-1.760e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[558] = 2.810e-13*exp(-1.760e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[559] = 3.320e-12*exp(-6.170e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[560] = 3.320e-12*exp(-6.170e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[561] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[562] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[563] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[564] = 3.120e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[565] = 3.120e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[566] = 1.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[567] = 1.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[568] = 1.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[569] = 1.650e-12*pow((T32),
        (+1.140e+00))*exp(-5.000e+01*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[570] = 1.650e-12*pow((T32),
        (+1.140e+00))*exp(-5.000e+01*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[571] = 1.650e-12*pow((T32),
        (+1.140e+00))*exp(-5.000e+01*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[572] = 1.690e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[573] = 1.690e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[574] = 1.690e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[575] = 1.690e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[576] = 8.000e-11*exp(-5.000e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[577] = 8.000e-11*exp(-5.000e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[578] = 8.000e-11*exp(-5.000e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[579] = 8.000e-11*exp(-5.000e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[580] = 1.500e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[581] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[582] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[583] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[584] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[585] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[586] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[587] = 1.500e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[588] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[589] = 7.500e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[590] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[591] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[592] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[593] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[594] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[595] = 8.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[596] = 8.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[597] = 2.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[598] = 2.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[599] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[600] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[601] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[602] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[603] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[604] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[605] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[606] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[607] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[608] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[609] = 8.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[610] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[611] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[612] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[613] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[614] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[615] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[616] = 3.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[617] = 3.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[618] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[619] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[620] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[621] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[622] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[623] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[624] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[625] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[626] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[627] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[628] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[629] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[630] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[631] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[632] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[633] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[634] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[635] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[636] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[637] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[638] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[639] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[640] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[641] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[642] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[643] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[644] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[645] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[646] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[647] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[648] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[649] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[650] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[651] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[652] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[653] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[654] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[655] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[656] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[657] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[658] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[659] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[660] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[661] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[662] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[663] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[664] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[665] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[666] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[667] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[668] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[669] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[670] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[671] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[672] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[673] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[674] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[675] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[676] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[677] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[678] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[679] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[680] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[681] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[682] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[683] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[684] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[685] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[686] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[687] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[688] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[689] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[690] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[691] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[692] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[693] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[694] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[695] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[696] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[697] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[698] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[699] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[700] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[701] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[702] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[703] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[704] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[705] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[706] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[707] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[708] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[709] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[710] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[711] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[712] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[713] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[714] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[715] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[716] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[717] = 7.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[718] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[719] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[720] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[721] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[722] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[723] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[724] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[725] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[726] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[727] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[728] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[729] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[730] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[731] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[732] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[733] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[734] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[735] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[736] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[737] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[738] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[739] = 5.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[740] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[741] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[742] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[743] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[744] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[745] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[746] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[747] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[748] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[749] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[750] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[751] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[752] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[753] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[754] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[755] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[756] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[757] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[758] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[759] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[760] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[761] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[762] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[763] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[764] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[765] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[766] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[767] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[768] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[769] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[770] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[771] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[772] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[773] = 1.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[774] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[775] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[776] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[777] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[778] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[779] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[780] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[781] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[782] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[783] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[784] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[785] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[786] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[787] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[788] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[789] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[790] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[791] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[792] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[793] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[794] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[795] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[796] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[797] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[798] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[799] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[800] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[801] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[802] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[803] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[804] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[805] = 3.333e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[806] = 3.333e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[807] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[808] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[809] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[810] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[811] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[812] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[813] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[814] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[815] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[816] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[817] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[818] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[819] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[820] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[821] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[822] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[823] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[824] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[825] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[826] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[827] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[828] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[829] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[830] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[831] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[832] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[833] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[834] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[835] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[836] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[837] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[838] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[839] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[840] = 7.667e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[841] = 7.667e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[842] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[843] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[844] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[845] = 7.667e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[846] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[847] = 7.667e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[848] = 7.667e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[849] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[850] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[851] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[852] = 7.667e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[853] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[854] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[855] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[856] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[857] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[858] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[859] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[860] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[861] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[862] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[863] = 3.830e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[864] = 3.830e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[865] = 3.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[866] = 3.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[867] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[868] = 3.300e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[869] = 3.300e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[870] = 3.300e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[871] = 3.300e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[872] = 3.300e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[873] = 3.300e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[874] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[875] = 6.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[876] = 6.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[877] = 6.400e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[878] = 1.000e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[879] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[880] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[881] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[882] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[883] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[884] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[885] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[886] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[887] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[888] = 2.890e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[889] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[890] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[891] = 6.260e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[892] = 6.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[893] = 6.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[894] = 6.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[895] = 6.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[896] = 6.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[897] = 3.125e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[898] = 3.125e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[899] = 4.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[900] = 2.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[901] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[902] = 1.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[903] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[904] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[905] = 6.600e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[906] = 6.600e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[907] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[908] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[909] = 6.600e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[910] = 6.600e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[911] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[912] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[913] = 1.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[914] = 1.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[915] = 3.100e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[916] = 3.100e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[917] = 6.900e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[918] = 6.900e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[919] = 6.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[920] = 6.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[921] = 6.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[922] = 6.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[923] = 6.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[924] = 6.900e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[925] = 4.430e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[926] = 4.430e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[927] = 4.430e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[928] = 4.430e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[929] = 4.430e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[930] = 4.430e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[931] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[932] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[933] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[934] = 1.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[935] = 8.450e-11*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[936] = 8.450e-11*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[937] = 8.450e-11*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[938] = 8.450e-11*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[939] = 2.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[940] = 2.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[941] = 2.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[942] = 2.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[943] = 2.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[944] = 1.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[945] = 1.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[946] = 1.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[947] = 1.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[948] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[949] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[950] = 9.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[951] = 9.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[952] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[953] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[954] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[955] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[956] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[957] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[958] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[959] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[960] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[961] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[962] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[963] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[964] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[965] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[966] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[967] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[968] = 2.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[969] = 2.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[970] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[971] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[972] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[973] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[974] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[975] = 3.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[976] = 9.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[977] = 9.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[978] = 4.550e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[979] = 4.550e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[980] = 6.400e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[981] = 6.400e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[982] = 6.400e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[983] = 6.400e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[984] = 2.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[985] = 2.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[986] = 1.640e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[987] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[988] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[989] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[990] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[991] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[992] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[993] = 9.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[994] = 9.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[995] = 4.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[996] = 4.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[997] = 4.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[998] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[999] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1000] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1001] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1002] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1003] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1004] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1005] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1006] = 1.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1007] = 1.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1008] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1009] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1010] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1011] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1012] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1013] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1014] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1015] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1016] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1017] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1018] = 8.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1019] = 8.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1020] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1021] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1022] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1023] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1024] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1025] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1026] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1027] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1028] = 2.330e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1029] = 2.330e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1030] = 2.330e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1031] = 2.330e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1032] = 4.200e-11*pow((T32),
        (-3.200e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1033] = 4.200e-11*pow((T32),
        (-3.200e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1034] = 2.690e-12*exp(-2.360e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1035] = 2.690e-12*exp(-2.360e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1036] = 2.690e-12*exp(-2.360e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1037] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1038] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1039] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1040] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1041] = 9.610e-13*exp(-1.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1042] = 9.610e-13*exp(-1.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1043] = 9.610e-13*exp(-1.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1044] = 9.610e-13*exp(-1.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1045] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1046] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1047] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1048] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1049] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1050] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1051] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1052] = 3.400e-11*pow((T32),
        (-3.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1053] = 4.980e-10*exp(-1.810e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1054] = 8.700e-11*exp(-2.260e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1055] = 1.000e-10*exp(-5.280e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1056] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1057] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1058] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1059] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1060] = 6.610e-11*exp(-5.160e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1061] = 6.610e-11*exp(-5.160e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1062] = 6.610e-11*exp(-5.160e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1063] = 6.610e-11*exp(-5.160e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<500.0) { k[1064] = 1.240e-10*pow((T32),
        (+2.600e-01));  }
        
    if (Tgas>10.0 && Tgas<500.0) { k[1065] = 1.240e-10*pow((T32),
        (+2.600e-01));  }
        
    if (Tgas>10.0 && Tgas<500.0) { k[1066] = 1.240e-10*pow((T32),
        (+2.600e-01));  }
        
    if (Tgas>10.0 && Tgas<500.0) { k[1067] = 1.240e-10*pow((T32),
        (+2.600e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1068] = 2.200e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1069] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1070] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1071] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1072] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1073] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1074] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1075] = 2.200e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1076] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1077] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1078] = 6.860e-14*pow((T32),
        (+2.800e+00))*exp(-1.950e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1079] = 6.860e-14*pow((T32),
        (+2.800e+00))*exp(-1.950e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1080] = 6.860e-14*pow((T32),
        (+2.800e+00))*exp(-1.950e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1081] = 6.860e-14*pow((T32),
        (+2.800e+00))*exp(-1.950e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1082] = 6.820e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1083] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1084] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1085] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1086] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1087] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1088] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1089] = 6.820e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1090] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1091] = 3.410e-12*pow((T32),
        (+1.600e+00))*exp(-9.720e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1092] = 6.190e-10*exp(-1.250e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1093] = 6.190e-10*exp(-1.250e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1094] = 6.190e-10*exp(-1.250e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1095] = 6.190e-10*exp(-1.250e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1096] = 9.300e-10*pow((T32),
        (-1.000e-01))*exp(-3.520e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1097] = 9.300e-10*pow((T32),
        (-1.000e-01))*exp(-3.520e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1098] = 3.600e-10*exp(-2.490e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1099] = 3.600e-10*exp(-2.490e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1100] = 1.050e-09*pow((T32),
        (-3.000e-01))*exp(-1.470e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1101] = 1.050e-09*pow((T32),
        (-3.000e-01))*exp(-1.470e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1102] = 1.050e-09*pow((T32),
        (-3.000e-01))*exp(-1.470e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1103] = 1.050e-09*pow((T32),
        (-3.000e-01))*exp(-1.470e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1104] = 2.410e-09*pow((T32),
        (-5.000e-01))*exp(-9.010e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1105] = 1.205e-09*pow((T32),
        (-5.000e-01))*exp(-9.010e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1106] = 1.205e-09*pow((T32),
        (-5.000e-01))*exp(-9.010e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1107] = 1.205e-09*pow((T32),
        (-5.000e-01))*exp(-9.010e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1108] = 1.205e-09*pow((T32),
        (-5.000e-01))*exp(-9.010e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1109] = 2.410e-09*pow((T32),
        (-5.000e-01))*exp(-9.010e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1110] = 2.940e-10*exp(-8.380e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1111] = 2.940e-10*exp(-8.380e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1112] = 5.600e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1113] = 5.600e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1114] = 5.600e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1115] = 5.600e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1116] = 7.210e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1117] = 7.210e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1118] = 7.210e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1119] = 7.210e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1120] = 2.420e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1121] = 2.420e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1122] = 2.420e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1123] = 2.420e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1124] = 2.510e-10*exp(-1.330e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1125] = 2.510e-10*exp(-1.330e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1126] = 9.220e-14*exp(-2.990e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1127] = 9.220e-14*exp(-2.990e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1128] = 4.000e-10*exp(-3.400e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1129] = 4.000e-10*exp(-3.400e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1130] = 6.640e-10*exp(-1.170e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1131] = 6.640e-10*exp(-1.170e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1132] = 6.640e-10*exp(-1.170e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1133] = 6.640e-10*exp(-1.170e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1134] = 3.320e-10*exp(-1.170e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1135] = 3.320e-10*exp(-1.170e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1136] = 3.750e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1137] = 3.750e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1138] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1139] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1140] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1141] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1142] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1143] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1144] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1145] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1146] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1147] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1148] = 3.750e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1149] = 3.750e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1150] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1151] = 1.875e-10*exp(-1.660e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1152] = 3.440e-13*pow((T32),
        (+2.670e+00))*exp(-3.160e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1153] = 3.440e-13*pow((T32),
        (+2.670e+00))*exp(-3.160e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1154] = 3.440e-13*pow((T32),
        (+2.670e+00))*exp(-3.160e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1155] = 3.440e-13*pow((T32),
        (+2.670e+00))*exp(-3.160e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1156] = 1.720e-13*pow((T32),
        (+2.670e+00))*exp(-3.160e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1157] = 1.720e-13*pow((T32),
        (+2.670e+00))*exp(-3.160e+03*invT);  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1158] = 8.400e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1159] = 8.400e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1160] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1161] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1162] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1163] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1164] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1165] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1166] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1167] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1168] = 8.400e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1169] = 8.400e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1170] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1171] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1172] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1173] = 4.200e-13*exp(-1.040e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1174] = 4.650e-10*exp(-1.660e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1175] = 4.650e-10*exp(-1.660e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1176] = 4.650e-10*exp(-1.660e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1177] = 4.650e-10*exp(-1.660e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1178] = 2.325e-10*exp(-1.660e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1179] = 2.325e-10*exp(-1.660e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1180] = 5.960e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1181] = 5.960e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1182] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1183] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1184] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1185] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1186] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1187] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1188] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1189] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1190] = 5.960e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1191] = 5.960e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1192] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1193] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1194] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1195] = 2.980e-11*exp(-7.780e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1196] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1197] = 1.660e-10*pow((T32),
        (-9.000e-02));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1198] = 1.660e-10*pow((T32),
        (-9.000e-02));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1199] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1200] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1201] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1202] = 3.000e-11*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1203] = 7.500e-11*pow((T32),
        (-1.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1204] = 7.500e-11*pow((T32),
        (-1.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1205] = 1.500e-11*exp(-3.680e+03*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1206] = 3.950e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1207] = 3.950e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1208] = 1.975e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1209] = 1.975e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1210] = 3.950e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1211] = 3.950e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1212] = 1.975e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1213] = 1.975e-11*pow((T32),
        (+1.670e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1214] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1215] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1216] = 1.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1217] = 1.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1218] = 2.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1219] = 1.700e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1220] = 1.700e-13;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1221] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1222] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1223] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1224] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1225] = 1.000e-13;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1226] = 2.000e-10*pow((T32),
        (-1.200e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1227] = 6.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1228] = 6.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1229] = 4.000e-11;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1230] = 6.600e-11;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1231] = 6.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1232] = 3.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1233] = 3.500e-11;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1234] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1235] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1236] = 6.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1237] = 8.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1238] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1239] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1240] = 2.350e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1241] = 2.350e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1242] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1243] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1244] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1245] = 1.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1246] = 1.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1247] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1248] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1249] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1250] = 1.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1251] = 1.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1252] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1253] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1254] = 2.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1255] = 2.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1256] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1257] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1258] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1259] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1260] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1261] = 6.667e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1262] = 3.334e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1263] = 3.334e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1264] = 1.333e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1265] = 1.333e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1266] = 6.667e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1267] = 3.334e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1268] = 3.334e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1269] = 1.333e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1270] = 1.333e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1271] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1272] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1273] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1274] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1275] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1276] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1277] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1278] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1279] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1280] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1281] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1282] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1283] = 8.540e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1284] = 4.270e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1285] = 4.270e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1286] = 8.540e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1287] = 8.540e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1288] = 2.847e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1289] = 1.424e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1290] = 1.424e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1291] = 5.693e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1292] = 5.693e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1293] = 2.847e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1294] = 1.424e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1295] = 1.424e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1296] = 5.693e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1297] = 5.693e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1298] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1299] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1300] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1301] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1302] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1303] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1304] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1305] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1306] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1307] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1308] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[1309] = 3.660e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1310] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1311] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1312] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1313] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1314] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1315] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1316] = 3.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1317] = 3.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1318] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1319] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1320] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1321] = 3.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1322] = 3.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1323] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1324] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1325] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1326] = 4.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1327] = 4.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1328] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1329] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1330] = 2.833e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1331] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1332] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1333] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1334] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1335] = 2.833e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1336] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1337] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1338] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1339] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1340] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1341] = 4.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1342] = 4.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1343] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1344] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1345] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1346] = 8.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1347] = 2.833e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1348] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1349] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1350] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1351] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1352] = 2.833e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1353] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1354] = 1.417e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1355] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1356] = 5.667e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1357] = 8.100e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1358] = 4.050e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1359] = 4.050e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1360] = 8.100e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1361] = 8.100e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1362] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1363] = 1.350e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1364] = 1.350e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1365] = 5.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1366] = 5.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1367] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1368] = 1.350e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1369] = 1.350e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1370] = 5.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1371] = 5.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1372] = 1.610e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1373] = 8.050e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1374] = 8.050e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1375] = 1.610e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1376] = 1.610e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1377] = 5.367e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1378] = 2.684e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1379] = 2.684e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1380] = 1.073e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1381] = 1.073e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1382] = 5.367e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1383] = 2.684e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1384] = 2.684e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1385] = 1.073e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1386] = 1.073e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1387] = 9.440e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1388] = 4.720e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1389] = 4.720e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1390] = 9.440e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1391] = 9.440e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1392] = 3.147e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1393] = 1.574e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1394] = 1.574e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1395] = 6.294e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1396] = 6.294e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1397] = 3.147e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1398] = 1.574e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1399] = 1.574e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1400] = 6.294e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1401] = 6.294e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1402] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1403] = 8.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1404] = 8.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1405] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1406] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1407] = 5.667e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1408] = 2.834e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1409] = 2.834e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1410] = 1.133e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1411] = 1.133e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1412] = 5.667e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1413] = 2.834e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1414] = 2.834e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1415] = 1.133e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1416] = 1.133e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1417] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1418] = 3.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1419] = 3.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1420] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1421] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1422] = 2.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1423] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1424] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1425] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1426] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1427] = 2.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1428] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1429] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1430] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1431] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1432] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1433] = 3.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1434] = 3.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1435] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1436] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1437] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1438] = 7.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1439] = 2.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1440] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1441] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1442] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1443] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1444] = 2.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1445] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1446] = 1.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1447] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1448] = 5.000e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1449] = 8.500e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1450] = 4.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1451] = 4.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1452] = 8.500e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1453] = 8.500e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1454] = 2.833e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1455] = 1.417e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1456] = 1.417e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1457] = 5.667e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1458] = 5.667e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1459] = 2.833e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1460] = 1.417e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1461] = 1.417e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1462] = 5.667e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1463] = 5.667e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1464] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1465] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1466] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1467] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1468] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1469] = 2.133e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1470] = 1.067e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1471] = 1.067e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1472] = 4.267e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1473] = 4.267e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1474] = 2.133e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1475] = 1.067e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1476] = 1.067e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1477] = 4.267e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1478] = 4.267e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1479] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1480] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1481] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1482] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1483] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1484] = 3.167e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1485] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1486] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1487] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1488] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1489] = 3.167e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1490] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1491] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1492] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1493] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1494] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1495] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1496] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1497] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1498] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1499] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1500] = 9.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1501] = 3.167e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1502] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1503] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1504] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1505] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1506] = 3.167e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1507] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1508] = 1.583e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1509] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1510] = 6.333e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1511] = 7.280e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1512] = 3.640e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1513] = 3.640e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1514] = 7.280e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1515] = 7.280e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1516] = 2.427e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1517] = 1.213e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1518] = 1.213e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1519] = 4.853e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1520] = 4.853e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1521] = 2.427e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1522] = 1.213e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1523] = 1.213e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1524] = 4.853e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1525] = 4.853e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1526] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1527] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1528] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1529] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1530] = 1.200e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1531] = 1.200e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1532] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1533] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1534] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1535] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1536] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1537] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1538] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1539] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1540] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1541] = 5.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1542] = 3.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1543] = 3.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1544] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1545] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1546] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1547] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1548] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1549] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1550] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1551] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1552] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1553] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1554] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1555] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1556] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1557] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1558] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1559] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1560] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1561] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1562] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1563] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1564] = 9.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1565] = 9.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1566] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1567] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1568] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1569] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1570] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1571] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1572] = 1.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1573] = 1.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1574] = 8.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1575] = 8.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1576] = 4.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1577] = 4.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1578] = 4.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1579] = 4.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1580] = 2.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1581] = 2.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1582] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1583] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1584] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1585] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1586] = 4.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1587] = 4.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1588] = 4.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1589] = 4.600e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1590] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1591] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1592] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1593] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1594] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1595] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1596] = 3.330e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1597] = 3.330e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1598] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1599] = 3.330e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1600] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1601] = 3.330e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1602] = 3.330e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1603] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1604] = 3.330e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1605] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1606] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1607] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1608] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1609] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1610] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1611] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1612] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1613] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1614] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1615] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1616] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1617] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1618] = 8.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1619] = 8.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1620] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1621] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1622] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1623] = 4.500e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1624] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1625] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1626] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1627] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1628] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1629] = 3.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1630] = 4.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1631] = 4.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1632] = 4.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1633] = 4.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1634] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1635] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1636] = 1.000e-09*exp(-8.500e+01*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1637] = 1.000e-09*exp(-8.500e+01*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1638] = 1.000e-09*exp(-8.500e+01*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1639] = 1.000e-09*exp(-8.500e+01*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1640] = 5.000e-10*exp(-8.500e+01*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1641] = 5.000e-10*exp(-8.500e+01*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1642] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1643] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1644] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1645] = 3.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1646] = 1.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1647] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1648] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1649] = 2.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1650] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1651] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1652] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1653] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1654] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1655] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1656] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1657] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1658] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1659] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1660] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1661] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1662] = 1.200e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1663] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1664] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1665] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1666] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1667] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1668] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1669] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1670] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1671] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1672] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1673] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1674] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1675] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1676] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1677] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1678] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1679] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1680] = 1.255e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1681] = 1.255e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1682] = 1.255e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1683] = 1.255e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1684] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1685] = 6.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1686] = 6.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1687] = 6.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1688] = 6.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1689] = 4.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1690] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1691] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1692] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1693] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1694] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1695] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1696] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1697] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1698] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1699] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1700] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1701] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1702] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1703] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1704] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1705] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1706] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1707] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1708] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1709] = 3.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1710] = 3.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1711] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1712] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1713] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1714] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1715] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1716] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1717] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1718] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1719] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1720] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1721] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1722] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1723] = 7.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1724] = 7.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1725] = 7.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1726] = 7.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1727] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1728] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1729] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1730] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1731] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1732] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1733] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1734] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1735] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1736] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1737] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1738] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1739] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1740] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1741] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1742] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1743] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1744] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1745] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1746] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1747] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1748] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1749] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1750] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1751] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1752] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1753] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1754] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1755] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1756] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1757] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1758] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1759] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1760] = 7.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1761] = 9.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1762] = 9.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1763] = 9.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1764] = 9.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1765] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1766] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1767] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1768] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1769] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1770] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1771] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1772] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1773] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1774] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1775] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1776] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1777] = 2.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1778] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1779] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1780] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1781] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1782] = 2.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1783] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1784] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1785] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1786] = 1.450e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1787] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1788] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1789] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1790] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1791] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1792] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1793] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1794] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1795] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1796] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1797] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1798] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1799] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1800] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1801] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1802] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1803] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1804] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1805] = 1.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1806] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1807] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1808] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1809] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1810] = 8.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1811] = 8.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1812] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1813] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1814] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1815] = 2.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1816] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1817] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1818] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1819] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1820] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1821] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1822] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1823] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1824] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1825] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1826] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1827] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1828] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1829] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1830] = 9.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1831] = 9.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1832] = 9.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1833] = 9.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1834] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1835] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1836] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1837] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1838] = 5.390e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1839] = 5.390e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1840] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1841] = 2.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1842] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1843] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1844] = 2.250e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1845] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1846] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1847] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1848] = 7.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1849] = 7.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1850] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1851] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1852] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1853] = 1.125e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1854] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1855] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1856] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1857] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1858] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1859] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1860] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1861] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1862] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1863] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1864] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1865] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1866] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1867] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1868] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1869] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1870] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1871] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1872] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1873] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1874] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1875] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1876] = 1.780e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1877] = 1.780e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1878] = 2.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1879] = 2.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1880] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1881] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1882] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1883] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1884] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1885] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1886] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1887] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1888] = 3.850e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1889] = 3.850e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1890] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1891] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1892] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1893] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1894] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1895] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1896] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1897] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1898] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1899] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1900] = 8.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1901] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1902] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1903] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1904] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1905] = 8.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1906] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1907] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1908] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1909] = 4.375e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1910] = 5.200e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1911] = 1.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1912] = 4.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1913] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1914] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1915] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1916] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1917] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1918] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1919] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1920] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1921] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1922] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1923] = 8.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1924] = 8.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1925] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1926] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1927] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1928] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1929] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1930] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1931] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1932] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1933] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1934] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1935] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1936] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1937] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1938] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1939] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1940] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1941] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1942] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1943] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1944] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1945] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1946] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1947] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1948] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1949] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1950] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1951] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1952] = 5.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1953] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1954] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1955] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1956] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1957] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1958] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1959] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1960] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1961] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1962] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1963] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1964] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1965] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1966] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1967] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1968] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1969] = 9.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1970] = 9.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1971] = 4.550e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1972] = 4.550e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1973] = 7.200e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1974] = 7.200e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1975] = 3.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1976] = 3.600e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1977] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1978] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1979] = 4.850e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1980] = 4.850e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1981] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1982] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1983] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1984] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1985] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1986] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1987] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1988] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1989] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1990] = 1.750e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1991] = 2.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1992] = 2.100e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1993] = 1.050e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1994] = 1.050e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1995] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1996] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1997] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1998] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[1999] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2000] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2001] = 1.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2002] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2003] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2004] = 1.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2005] = 1.000e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2006] = 6.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2007] = 6.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2008] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2009] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2010] = 6.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2011] = 6.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2012] = 6.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2013] = 6.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2014] = 8.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2015] = 8.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2016] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2017] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2018] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2019] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2020] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2021] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2022] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2023] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2024] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2025] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2026] = 1.067e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2027] = 1.067e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2028] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2029] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2030] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2031] = 1.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2032] = 7.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2033] = 7.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2034] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2035] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2036] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2037] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2038] = 7.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2039] = 7.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2040] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2041] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2042] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2043] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2044] = 2.000e-11*pow((T32),
        (+4.400e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2045] = 2.000e-11*pow((T32),
        (+4.400e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2046] = 6.590e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2047] = 6.590e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2048] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2049] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2050] = 6.000e-11*pow((T32),
        (-1.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2051] = 9.000e-11*pow((T32),
        (-1.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2052] = 4.700e-11*pow((T32),
        (-3.400e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2053] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2054] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2055] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2056] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2057] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2058] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2059] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2060] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2061] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2062] = 2.000e-10;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[2063] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2064] = 6.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2065] = 4.800e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2066] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2067] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2068] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2069] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2070] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2071] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2072] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2073] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2074] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2075] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2076] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2077] = 8.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2078] = 9.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2079] = 9.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2080] = 9.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2081] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2082] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2083] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2084] = 1.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2085] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2086] = 3.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2087] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2088] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2089] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2090] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2091] = 7.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2092] = 7.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2093] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2094] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2095] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2096] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2097] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2098] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2099] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2100] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2101] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2102] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2103] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2104] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2105] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2106] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2107] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2108] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2109] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2110] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2111] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2112] = 2.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2113] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2114] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2115] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2116] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2117] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2118] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2119] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2120] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2121] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2122] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2123] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2124] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2125] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2126] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2127] = 2.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2128] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2129] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2130] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2131] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2132] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2133] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2134] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2135] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2136] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2137] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2138] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2139] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2140] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2141] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2142] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2143] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2144] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2145] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2146] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2147] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2148] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2149] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2150] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2151] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2152] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2153] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2154] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2155] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2156] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2157] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2158] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2159] = 2.300e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2160] = 1.900e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2161] = 1.900e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2162] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2163] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2164] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2165] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2166] = 4.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2167] = 1.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2168] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2169] = 3.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2170] = 3.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2171] = 3.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2172] = 3.700e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2173] = 6.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2174] = 6.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2175] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2176] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2177] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2178] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2179] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2180] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2181] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2182] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2183] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2184] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2185] = 7.000e-10*exp(-2.320e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2186] = 7.000e-10*exp(-2.320e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2187] = 3.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2188] = 3.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2189] = 1.400e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2190] = 1.400e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2191] = 1.400e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2192] = 1.400e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2193] = 1.200e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2194] = 1.200e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2195] = 1.200e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2196] = 1.200e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2197] = 1.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2198] = 1.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2199] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2200] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2201] = 1.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2202] = 1.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2203] = 1.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2204] = 1.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2205] = 5.560e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2206] = 5.560e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2207] = 4.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2208] = 4.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2209] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2210] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2211] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2212] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2213] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2214] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2215] = 2.780e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2216] = 2.780e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2217] = 2.780e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2218] = 2.780e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2219] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2220] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2221] = 3.600e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2222] = 3.600e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2223] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2224] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2225] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2226] = 9.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2227] = 9.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2228] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2229] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2230] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2231] = 2.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2232] = 2.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2233] = 2.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2234] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2235] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2236] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2237] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2238] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2239] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2240] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2241] = 1.700e-12;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2242] = 3.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2243] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2244] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2245] = 9.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2246] = 3.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2247] = 3.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2248] = 3.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2249] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2250] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2251] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2252] = 1.600e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2253] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2254] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2255] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2256] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2257] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2258] = 7.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2259] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2260] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2261] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2262] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2263] = 1.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2264] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2265] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2266] = 6.500e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2267] = 8.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2268] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2269] = 6.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2270] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2271] = 8.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2272] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2273] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2274] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2275] = 7.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2276] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2277] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2278] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2279] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2280] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2281] = 6.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2282] = 7.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2283] = 7.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2284] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2285] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2286] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2287] = 9.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2288] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2289] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2290] = 8.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2291] = 8.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2292] = 8.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2293] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2294] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2295] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2296] = 8.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2297] = 6.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2298] = 5.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2299] = 5.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2300] = 9.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2301] = 9.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2302] = 9.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2303] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2304] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2305] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2306] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2307] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2308] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2309] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2310] = 2.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2311] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2312] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2313] = 4.340e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2314] = 6.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2315] = 6.700e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2316] = 8.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2317] = 7.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2318] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2319] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2320] = 2.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2321] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2322] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2323] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2324] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2325] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2326] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2327] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2328] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2329] = 3.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2330] = 3.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2331] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2332] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2333] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2334] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2335] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2336] = 1.700e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2337] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2338] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2339] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2340] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2341] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2342] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2343] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2344] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2345] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2346] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2347] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2348] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2349] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2350] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2351] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2352] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2353] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2354] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2355] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2356] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2357] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2358] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2359] = 1.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2360] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2361] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2362] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2363] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2364] = 6.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2365] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2366] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2367] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2368] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2369] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2370] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2371] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2372] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2373] = 1.320e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2374] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2375] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2376] = 9.620e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2377] = 2.040e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2378] = 2.040e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2379] = 2.040e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2380] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2381] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2382] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2383] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2384] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2385] = 1.800e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2386] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2387] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2388] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2389] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2390] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2391] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2392] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2393] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2394] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2395] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2396] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2397] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2398] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2399] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2400] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2401] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2402] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2403] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2404] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2405] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2406] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2407] = 8.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2408] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2409] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2410] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2411] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2412] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2413] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2414] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2415] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2416] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2417] = 7.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2418] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2419] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2420] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2421] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2422] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2423] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2424] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2425] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2426] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2427] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2428] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2429] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2430] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2431] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2432] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2433] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2434] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2435] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2436] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2437] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2438] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2439] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2440] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2441] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2442] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2443] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2444] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2445] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2446] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2447] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2448] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2449] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2450] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2451] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2452] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2453] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2454] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2455] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2456] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2457] = 3.900e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2458] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2459] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2460] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2461] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2462] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2463] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2464] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2465] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2466] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2467] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2468] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2469] = 3.830e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2470] = 3.830e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2471] = 9.600e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2472] = 9.600e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2473] = 9.600e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2474] = 9.600e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2475] = 9.600e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2476] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2477] = 3.300e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2478] = 1.210e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2479] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2480] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2481] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2482] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2483] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2484] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2485] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2486] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2487] = 3.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2488] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2489] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2490] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2491] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2492] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2493] = 4.400e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2494] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2495] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2496] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2497] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2498] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2499] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2500] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2501] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2502] = 4.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2503] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2504] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2505] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2506] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2507] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2508] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2509] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2510] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2511] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2512] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2513] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2514] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2515] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2516] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2517] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2518] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2519] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2520] = 5.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2521] = 1.100e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2522] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2523] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2524] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2525] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2526] = 3.700e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2527] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2528] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2529] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2530] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2531] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2532] = 1.140e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2533] = 4.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2534] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2535] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2536] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2537] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2538] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2539] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2540] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2541] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2542] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2543] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2544] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2545] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2546] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2547] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2548] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2549] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2550] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2551] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2552] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2553] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2554] = 4.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2555] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2556] = 7.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2557] = 5.200e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2558] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2559] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2560] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2561] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2562] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2563] = 1.300e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2564] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2565] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2566] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2567] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2568] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2569] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2570] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2571] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2572] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2573] = 4.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2574] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2575] = 3.700e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2576] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2577] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2578] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2579] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2580] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2581] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2582] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2583] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2584] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2585] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2586] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2587] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2588] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2589] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2590] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2591] = 4.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2592] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2593] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2594] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2595] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2596] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2597] = 1.500e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2598] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2599] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2600] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2601] = 2.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2602] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2603] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2604] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2605] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2606] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2607] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2608] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2609] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2610] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2611] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2612] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2613] = 3.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2614] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2615] = 3.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2616] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2617] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2618] = 4.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2619] = 3.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2620] = 3.800e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2621] = 1.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2622] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2623] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2624] = 2.200e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2625] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2626] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2627] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2628] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2629] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2630] = 1.050e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2631] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2632] = 3.100e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2633] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2634] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2635] = 4.300e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2636] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2637] = 3.600e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2638] = 1.700e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2639] = 1.700e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2640] = 2.500e-18;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2641] = 4.000e-16*pow((T32),
        (-2.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2642] = 4.000e-16*pow((T32),
        (-2.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2643] = 4.000e-16*pow((T32),
        (-2.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2644] = 4.000e-16*pow((T32),
        (-2.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2645] = 4.000e-16*pow((T32),
        (-2.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2646] = 1.000e-20*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2647] = 1.000e-20*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2648] = 2.000e-20*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2649] = 2.000e-20*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2650] = 2.000e-20*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2651] = 2.000e-20*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2652] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2653] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2654] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2655] = 1.000e-17;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2656] = 2.100e-19;  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[2657] = 2.000e-20*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[2658] = 2.000e-20*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[2659] = 2.000e-20*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[2660] = 2.000e-20*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<298.0) { k[2661] = 2.000e-20*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2662] = 9.900e-19*pow((T32),
        (-3.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2663] = 9.900e-19*pow((T32),
        (-3.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2664] = 4.900e-20*pow((T32),
        (+1.580e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2665] = 3.300e-16*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2666] = 4.000e-18*pow((T32),
        (-2.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2667] = 4.000e-18*pow((T32),
        (-2.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2668] = 4.000e-18*pow((T32),
        (-2.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2669] = 4.000e-18*pow((T32),
        (-2.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2670] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2671] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2672] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2673] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2674] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2675] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2676] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2677] = 1.000e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2678] = 1.000e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2679] = 1.000e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2680] = 1.000e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2681] = 1.000e-13;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2682] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2683] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2684] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2685] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2686] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2687] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2688] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2689] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2690] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2691] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2692] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2693] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2694] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2695] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2696] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2697] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2698] = 1.000e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2699] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2700] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2701] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2702] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2703] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2704] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2705] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2706] = 5.000e-11;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2707] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2708] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2709] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2710] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2711] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2712] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2713] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2714] = 1.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2715] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2716] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2717] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2718] = 2.200e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2719] = 1.900e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2720] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2721] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2722] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2723] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2724] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2725] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2726] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2727] = 7.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2728] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2729] = 1.300e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2730] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2731] = 5.000e-10;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2732] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2733] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2734] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2735] = 1.400e-09;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2736] = 3.000e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2737] = 3.000e-16*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2738] = 3.000e-16*pow((T32),
        (+1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2739] = 1.500e-15;  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2740] = 8.840e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2741] = 7.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2742] = 7.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2743] = 3.380e-07*pow((T32),
        (-5.500e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2744] = 2.750e-07*pow((T32),
        (-5.500e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2745] = 2.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2746] = 2.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2747] = 2.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2748] = 2.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2749] = 2.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2750] = 2.250e-07*pow((T32),
        (-4.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2751] = 2.250e-07*pow((T32),
        (-4.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2752] = 2.250e-07*pow((T32),
        (-4.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2753] = 2.250e-07*pow((T32),
        (-4.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2754] = 2.250e-07*pow((T32),
        (-4.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2755] = 2.250e-07*pow((T32),
        (-4.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2756] = 3.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2757] = 3.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2758] = 1.800e-07*pow((T32),
        (-3.900e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2759] = 1.180e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2760] = 1.180e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2761] = 4.100e-07*pow((T32),
        (-1.000e+00));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2762] = 1.950e-07*pow((T32),
        (-7.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2763] = 6.300e-09*pow((T32),
        (-4.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2764] = 6.300e-09*pow((T32),
        (-4.800e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2765] = 1.160e-07*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2766] = 1.160e-07*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2767] = 1.050e-07*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2768] = 1.050e-07*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2769] = 4.800e-08*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2770] = 4.800e-08*pow((T32),
        (-7.600e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2771] = 1.500e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2772] = 1.500e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2773] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2774] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2775] = 7.700e-08*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2776] = 7.700e-08*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2777] = 7.700e-08*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2778] = 1.600e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2779] = 1.600e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2780] = 1.600e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2781] = 1.600e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2782] = 4.000e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2783] = 4.000e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2784] = 4.000e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[2785] = 3.800e-07*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<1000.0) { k[2786] = 2.000e-08*pow((T32),
        (-6.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2787] = 4.200e-07*pow((T32),
        (-7.500e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2788] = 3.900e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2789] = 3.900e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2790] = 3.900e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2791] = 8.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2792] = 8.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2793] = 8.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2794] = 8.600e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2795] = 3.050e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2796] = 3.050e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2797] = 3.050e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2798] = 4.870E-08*pow((T32),
        (1.600E-01))*exp(+1.010E+00*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2799] = 3.560E-08*pow((T32),
        (-7.300E-01))*exp(-9.800E-01*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2800] = 2.160E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2801] = 2.160E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2802] = 4.380E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2803] = 4.380E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2804] = 4.380E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2805] = 4.380E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2806] = 2.510E-08*pow((T32),
        (1.600E-01))*exp(+1.010E+00*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2807] = 9.200E-09*pow((T32),
        (-7.300E-01))*exp(-9.800E-01*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2808] = 9.200E-09*pow((T32),
        (-7.300E-01))*exp(-9.800E-01*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2809] = 5.400E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2810] = 2.700E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2811] = 2.700E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2812] = 1.200E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2813] = 1.200E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2814] = 4.200E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2815] = 4.200E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2816] = 1.200E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2817] = 1.200E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2818] = 4.200E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2819] = 4.200E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2820] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2821] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2822] = 2.800e-07*pow((T32),
        (-6.900e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2823] = 2.800e-07*pow((T32),
        (-6.900e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2824] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2825] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2826] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2827] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2828] = 2.000e-07*pow((T32),
        (-7.500e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2829] = 2.000e-07*pow((T32),
        (-7.500e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2830] = 9.000e-08*pow((T32),
        (-5.100e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2831] = 9.000e-08*pow((T32),
        (-5.100e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2832] = 1.000e-08*pow((T32),
        (-5.100e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2833] = 1.000e-08*pow((T32),
        (-5.100e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2834] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2835] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2836] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2837] = 2.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2838] = 1.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2839] = 1.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2840] = 1.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2841] = 1.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2842] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2843] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2844] = 3.000e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2845] = 4.400e-12*pow((T32),
        (-6.100e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2846] = 3.500e-12*pow((T32),
        (-7.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2847] = 3.500e-12*pow((T32),
        (-7.000e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2848] = 4.500e-12*pow((T32),
        (-6.700e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2849] = 3.800e-12*pow((T32),
        (-6.200e-01));  }
        
    if (Tgas>10.0 && Tgas<280.0) { k[2850] = 3.400e-12*pow((T32),
        (-6.300e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2851] =
        2.100e-09*exp(-4.050e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2852] =
        2.100e-09*exp(-4.910e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2853] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2854] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2855] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2856] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2857] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2858] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2859] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2860] =
        1.000e-09*exp(-6.000e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2861] =
        1.000e-09*exp(-4.633e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2862] =
        1.000e-09*exp(-5.135e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2863] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2864] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2865] =
        1.000e-09*exp(-4.300e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2866] =
        1.000e-09*exp(-4.720e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2867] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2868] = 0.500e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2869] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2870] =
        1.000e-09*exp(-6.600e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2871] =
        1.000e-09*exp(-6.550e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2872] =
        1.980e-09*exp(-8.600e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2873] = 1.320e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2874] = 2.200e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2875] =
        1.980e-09*exp(-1.705e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2876] = 7.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2877] = 1.400e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2878] = 1.400e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2879] = 2.100e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2880] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2881] =
        1.000e-09*exp(-4.100e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2882] = 2.100e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2883] = 2.100e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2884] =
        1.000e-09*exp(-4.640e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2885] =
        1.000e-09*exp(-6.345e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2886] = 1.050e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2887] = 5.250e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2888] = 1.050e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2889] = 5.250e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2890] = 0.660e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2891] = 0.330e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2892] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2893] =
        1.000e-09*exp(-6.320e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2894] =
        0.500e-09*exp(-5.784e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2895] =
        0.500e-09*exp(-5.455e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2896] = 7.000e-18*pow((T32),
        (+1.800e+00))*exp(-1.000e-99*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2897] = 1.200e-17*pow((T32),
        (+1.800e+00))*exp(-1.000e-99*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2898] =
        1.000e-09*exp(-1.540e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2899] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2900] =
        1.450e-11*exp(+6.900e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2901] =
        4.090e-11*exp(+7.100e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2902] =
        1.690e-09*exp(-5.230e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2903] =
        1.480e-11*exp(+6.200e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2904] =
        1.690e-11*exp(+6.400e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2905] =
        6.870e-11*exp(+8.500e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2906] =
        7.440e-11*exp(-1.100e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2907] =
        8.940e-11*exp(+1.000e+00*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2908] =
        8.370e-11*exp(+6.100e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2909] =
        4.490e-11*exp(+2.300e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2910] =
        4.000e-10*exp(-2.170e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2911] =
        5.340e-10*exp(-6.890e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2912] = 6.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2913] = 6.500e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2914] = 2.160E-08*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2915] = 5.400E-09*pow((T32),
        (-5.000E-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2916] = 2.070e-16*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2917] = 1.048e-16*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2918] = 4.900e-17*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2919] = 4.700e-17*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2920] = 4.400e-17*pow((T32),
        (+5.000e-01))*user_GtoDN;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2921] = 5.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2922] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2923] = 9.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2924] = 1.800e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2925] = 1.125e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2926] = 2.250e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2927] = 1.600e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2928] = 3.200e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2929] = 5.083e-15*pow((T32),
        (+5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2930] = 4.200e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2931] = 7.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2932] = 4.200e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2933] = 7.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2934] = 7.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2935] = 2.100e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2936] =
        2.210e-10*exp(-3.792e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2937] =
        1.770e-09*exp(-2.252e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2938] =
        3.000e-10*exp(-2.867e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2939] =
        2.770e-10*exp(-2.297e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2940] =
        2.240e-10*exp(-1.448e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2941] =
        1.500e-10*exp(-1.820e+02*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2942] = 7.750e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2943] =
        9.160e-11*exp(-1.550e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2944] =
        4.610e-10*exp(+2.900e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2945] =
        4.750e-10*exp(-5.400e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2946] =
        6.180e-10*exp(+7.700e-01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2947] =
        5.370e-11*exp(-1.520e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2948] =
        2.510e-11*exp(-9.950e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2949] =
        7.740e-11*exp(-1.530e+01*invT);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2950] = 3.333e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2951] = 5.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2952] = 6.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2953] = 9.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2954] = 7.500e-11;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2955] = 1.125e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2956] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2957] = 1.000e-09;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2958] = 1.000e-17;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2959] = 4.270e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2960] = 4.270e-10*pow((T32),
        (-2.100e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2961] = 9.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2962] = 9.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2963] = 4.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2964] = 4.250e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2965] = 4.050e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2966] = 4.050e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2967] = 6.000e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2968] = 8.050e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2969] = 8.050e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2970] = 6.500e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2971] = 3.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2972] = 3.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2973] = 4.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2974] = 4.250e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2975] = 3.200e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2976] = 3.200e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2977] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2978] = 4.750e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2979] = 3.640e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2980] = 3.640e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2981] = 1.067e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2982] = 1.600e-10;  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2983] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2984] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2985] =
        1.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2986] =
        1.000e-08*exp(-1.800e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2987] =
        3.950e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>-9999.0 && Tgas<9999.0) { k[2988] =
        3.950e-09*exp(-2.300e+00*user_Av);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[2989] = 4.660e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[2990] = 4.660e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2991] = 1.720e-07*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2992] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2993] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2994] = 1.720e-07*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[2995] = 3.210e-01*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[2996] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2997] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[2998] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[2999] = 1.000e+00*9.850e-10*(0.62e0 +
        0.4767e0*4.400e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3000] = 1.000e+00*8.210e-10*(0.62e0 +
        0.4767e0*5.500e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3001] = 1.000e+00*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3002] = 3.330e-01*1.120e-09*(0.62e0 +
        0.4767e0*3.580e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3003] = 8.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3004] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3005] = 6.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3006] = 3.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3007] = 1.000e+00*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3008] = 1.000e+00*8.220e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3009] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3010] = 2.500e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3011] = 7.500e-12*exp(-1.700e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3012] = 1.000e+00*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3013] = 1.430e-01*1.010e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3014] = 4.620e-01*9.710e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3015] = 4.640e-01*9.390e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3016] = 9.250e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3017] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3018] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3019] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3020] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3021] = 1.520e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3022] = 4.570e-10*exp(-2.070e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3023] = 3.000e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3024] = 3.000e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3025] = 2.600e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3026] = 1.100e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3027] = 1.500e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3028] = 4.500e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3029] = 4.200e-09*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3030] = 1.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3031] = 3.500e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3032] = 1.170e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3033] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3034] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3035] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3036] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3037] = 3.500e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3038] = 3.500e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3039] = 1.170e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3040] = 1.170e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3041] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3042] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3043] = 1.170e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3044] = 1.170e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3045] = 3.500e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3046] = 3.500e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3047] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3048] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3049] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3050] = 2.330e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3051] = 1.170e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3052] = 3.500e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3053] = 4.660e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3054] = 4.660e-01*2.070e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3055] = 4.310e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3056] = 1.440e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3057] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3058] = 4.310e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3059] = 1.440e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3060] = 3.830e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3061] = 1.920e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3062] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3063] = 3.830e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3064] = 1.920e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3065] = 7.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3066] = 1.280e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3067] = 2.550e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3068] = 1.910e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3069] = 1.910e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3070] = 7.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3071] = 7.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3072] = 1.910e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3073] = 1.910e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3074] = 2.870e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3075] = 9.570e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3076] = 7.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3077] = 2.240e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3078] = 3.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3079] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3080] = 2.240e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3081] = 3.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3082] = 2.870e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3083] = 2.870e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3084] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3085] = 2.870e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3086] = 2.870e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3087] = 9.580e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3088] = 1.340e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3089] = 2.870e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3090] = 8.620e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3091] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3092] = 3.830e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3093] = 7.670e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3094] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3095] = 1.910e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3096] = 1.910e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3097] = 1.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3098] = 2.550e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3099] = 1.280e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3100] = 9.570e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3101] = 2.870e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3102] = 1.530e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3103] = 1.910e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3104] = 1.910e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3105] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3106] = 4.470e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3107] = 7.030e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3108] = 1.150e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3109] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3110] = 5.750e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3111] = 1.340e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3112] = 9.580e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3113] = 3.210e-01*pow((T32),
        (8.450e-10))*exp(-5.410e+00*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3114] = 3.210e-01*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3115] = 3.210e-01*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3116] = 3.210e-01*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3117] = 3.210e-01*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3118] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3119] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3120] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3121] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3122] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3123] = 4.570e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3124] = 4.570e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3125] = 1.520e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3126] = 1.520e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3127] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3128] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3129] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3130] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3131] = 4.570e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3132] = 1.520e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3133] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3134] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3135] = 1.520e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3136] = 4.570e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3137] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3138] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3139] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3140] = 3.050e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3141] = 1.520e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3142] = 1.520e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3143] = 4.570e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3144] = 4.570e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3145] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3146] = 6.100e-10;  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3147] = 1.000e+00*9.850e-10*(0.62e0 +
        0.4767e0*4.400e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3148] = 1.000e+00*9.850e-10*(0.62e0 +
        0.4767e0*4.400e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3149] = 1.000e+00*9.850e-10*(0.62e0 +
        0.4767e0*4.400e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3150] = 1.000e+00*9.850e-10*(0.62e0 +
        0.4767e0*4.400e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3151] = 1.000e+00*9.850e-10*(0.62e0 +
        0.4767e0*4.400e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3152] = 1.000e+00*8.210e-10*(0.62e0 +
        0.4767e0*5.500e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3153] = 1.000e+00*8.210e-10*(0.62e0 +
        0.4767e0*5.500e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3154] = 1.000e+00*8.210e-10*(0.62e0 +
        0.4767e0*5.500e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3155] = 1.000e+00*8.210e-10*(0.62e0 +
        0.4767e0*5.500e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3156] = 1.000e+00*8.210e-10*(0.62e0 +
        0.4767e0*5.500e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3157] = 7.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3158] = 2.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3159] = 5.000e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3160] = 5.000e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3161] = 7.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3162] = 2.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3163] = 5.000e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3164] = 5.000e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3165] = 2.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3166] = 7.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3167] = 5.000e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3168] = 5.000e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3169] = 2.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3170] = 7.500e-01*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3171] = 1.000e+00*9.260e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3172] = 3.330e-01*1.120e-09*(0.62e0 +
        0.4767e0*3.580e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3173] = 3.330e-01*1.120e-09*(0.62e0 +
        0.4767e0*3.580e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3174] = 3.330e-01*1.120e-09*(0.62e0 +
        0.4767e0*3.580e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3175] = 3.330e-01*1.120e-09*(0.62e0 +
        0.4767e0*3.580e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3176] = 3.330e-01*1.120e-09*(0.62e0 +
        0.4767e0*3.580e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3177] = 1.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3178] = 4.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3179] = 2.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3180] = 3.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3181] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3182] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3183] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3184] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3185] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3186] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3187] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3188] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3189] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3190] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3191] = 1.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3192] = 1.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3193] = 1.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3194] = 1.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3195] = 1.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3196] = 1.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3197] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3198] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3199] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3200] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3201] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3202] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3203] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3204] = 7.500e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3205] = 5.250e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3206] = 2.250e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3207] = 3.750e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3208] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3209] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3210] = 1.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3211] = 1.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3212] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3213] = 2.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3214] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3215] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3216] = 3.330e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3217] = 6.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3218] = 3.330e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3219] = 6.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3220] = 2.500e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3221] = 7.500e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3222] = 2.500e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3223] = 7.500e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3224] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3225] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3226] = 1.330e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3227] = 1.670e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3228] = 2.330e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3229] = 6.670e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3230] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3231] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3232] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3233] = 5.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3234] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3235] = 2.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3236] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3237] = 3.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3238] = 2.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3239] = 1.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3240] = 1.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3241] = 3.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3242] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3243] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3244] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3245] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3246] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3247] = 5.000e-02*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3248] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3249] = 1.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3250] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3251] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3252] = 1.830e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3253] = 1.170e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3254] = 1.170e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3255] = 1.830e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3256] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3257] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3258] = 3.170e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3259] = 2.830e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3260] = 4.170e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3261] = 1.830e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3262] = 5.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3263] = 4.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3264] = 2.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3265] = 7.500e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3266] = 4.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3267] = 6.000e-01*1.730e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3268] = 1.000e+00*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3269] = 1.000e+00*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3270] = 1.000e+00*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3271] = 1.000e+00*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3272] = 1.000e+00*8.450e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3273] = 1.000e+00*8.220e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3274] = 1.000e+00*8.220e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3275] = 1.000e+00*8.220e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3276] = 1.000e+00*8.220e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3277] = 1.000e+00*8.220e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3278] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3279] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3280] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3281] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3282] = 1.000e+00*8.340e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3283] = 1.670e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3284] = 1.670e-12*exp(-1.700e+02*invT);
    }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3285] = 6.670e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3286] = 6.670e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3287] = 1.670e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3288] = 1.670e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3289] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3290] = 5.000e-12;  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3291] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3292] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3293] = 1.670e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3294] = 6.670e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3295] = 1.670e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3296] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3297] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3298] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3299] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3300] = 1.670e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3301] = 6.670e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3302] = 1.670e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3303] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3304] = 5.000e-01*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3305] = 1.000e+00*1.230e-09*(0.62e0 +
        0.4767e0*3.330e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3306] = 1.430e-01*1.010e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3307] = 1.430e-01*1.010e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3308] = 1.430e-01*1.010e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3309] = 1.430e-01*1.010e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3310] = 1.430e-01*1.010e-09*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3311] = 4.620e-01*9.710e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3312] = 4.620e-01*9.710e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3313] = 4.620e-01*9.710e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3314] = 4.620e-01*9.710e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3315] = 4.620e-01*9.710e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3316] = 4.640e-01*9.390e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3317] = 4.640e-01*9.390e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3318] = 4.640e-01*9.390e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3319] = 4.640e-01*9.390e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3320] = 4.640e-01*9.390e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3321] = 6.940e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3322] = 2.310e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3323] = 4.630e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3324] = 4.630e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3325] = 6.940e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3326] = 2.310e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3327] = 4.630e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3328] = 4.630e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3329] = 2.310e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3330] = 6.940e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3331] = 4.630e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3332] = 4.630e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3333] = 2.310e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3334] = 6.940e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3335] = 9.250e-01*9.540e-10*(0.62e0 +
        0.4767e0*5.410e+00*sqrt(3e2*invT));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3336] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3337] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3338] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3339] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3340] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3341] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3342] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3343] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3344] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3345] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3346] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3347] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3348] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3349] = 2.510e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3350] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3351] = 1.880e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3352] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3353] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3354] = 1.880e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3355] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3356] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3357] = 6.270e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3358] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3359] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3360] = 6.270e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3361] = 6.270e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3362] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3363] = 1.250e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3364] = 6.270e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3365] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3366] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3367] = 1.880e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3368] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3369] = 9.400e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3370] = 1.880e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3371] = 3.760e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3372] = 1.020e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3373] = 2.030e-10*exp(-2.070e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3374] = 3.050e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3375] = 2.550e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3376] = 7.650e-11*exp(-2.070e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3377] = 4.070e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3378] = 5.100e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3379] = 5.100e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3380] = 3.050e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3381] = 1.520e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3382] = 1.520e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3383] = 7.620e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3384] = 2.290e-10*exp(-2.070e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3385] = 3.050e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3386] = 5.100e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3387] = 5.100e-11*exp(-2.070e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3388] = 4.070e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3389] = 6.800e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3390] = 3.400e-11*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3391] = 3.050e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3392] = 1.190e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3393] = 1.860e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3394] = 3.560e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3395] = 2.540e-10*exp(-2.050e+04*invT);
    }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3396] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3397] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3398] = 1.200e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3399] = 1.200e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3400] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3401] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3402] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3403] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3404] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3405] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3406] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3407] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3408] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3409] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3410] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3411] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3412] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3413] = 1.200e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3414] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3415] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3416] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3417] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3418] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3419] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3420] = 1.200e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3421] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3422] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3423] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3424] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3425] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3426] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3427] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3428] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3429] = 3.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3430] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3431] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3432] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3433] = 9.000e-12*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3434] = 1.200e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3435] = 1.200e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3436] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3437] = 1.800e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3438] = 3.000e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<800.0) { k[3439] = 3.000e-11*pow((T32),
        (5.000e-01))*exp(-5.200e+04*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3440] = 8.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3441] = 1.730e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3442] = 1.730e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3443] = 8.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3444] = 2.600e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3445] = 7.330e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3446] = 3.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3447] = 3.670e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3448] = 7.330e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3449] = 1.100e-07*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3450] = 1.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3451] = 1.000e-08*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3452] = 4.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3453] = 4.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3454] = 1.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3455] = 1.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3456] = 3.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3457] = 3.000e-08*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3458] = 3.730e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3459] = 9.350e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3460] = 9.350e-10*pow((T32),
        (-5.000e-01))*exp(-1.700e+02*invT);  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3461] = 9.350e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3462] = 9.350e-10*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3463] = 3.730e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3464] = 2.800e-09*pow((T32),
        (-5.000e-01));  }
        
    if (Tgas>10.0 && Tgas<300.0) { k[3465] = 2.800e-09*pow((T32),
        (-5.000e-01));  }
        
    
        // clang-format on

    return NAUNET_SUCCESS;
}