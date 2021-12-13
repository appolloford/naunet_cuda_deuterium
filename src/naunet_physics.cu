#include <math.h>

#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

// clang-format off
__device__ __host__ double GetMantleDens(double *y) {
    return 0.0;
}

__device__ __host__ double GetMu(double *y) {
    return (y[IDX_CI]*12.0 + y[IDX_CII]*12.0 + y[IDX_CM]*12.0 + y[IDX_C2I]*24.0 +
        y[IDX_C2II]*24.0 + y[IDX_C2DI]*26.0 + y[IDX_C2DII]*26.0 +
        y[IDX_C2HI]*25.0 + y[IDX_C2HII]*25.0 + y[IDX_C2NI]*38.0 +
        y[IDX_C2NII]*38.0 + y[IDX_C2OII]*40.0 + y[IDX_C3I]*36.0 +
        y[IDX_C3II]*36.0 + y[IDX_CCOI]*40.0 + y[IDX_CDI]*14.0 + y[IDX_CDII]*14.0
        + y[IDX_CD2I]*16.0 + y[IDX_CD2II]*16.0 + y[IDX_CHI]*13.0 +
        y[IDX_CHII]*13.0 + y[IDX_CH2I]*14.0 + y[IDX_CH2II]*14.0 +
        y[IDX_CHDI]*15.0 + y[IDX_CHDII]*15.0 + y[IDX_CNI]*26.0 +
        y[IDX_CNII]*26.0 + y[IDX_CNM]*26.0 + y[IDX_CNCII]*38.0 + y[IDX_COI]*28.0
        + y[IDX_COII]*28.0 + y[IDX_CO2I]*44.0 + y[IDX_CO2II]*44.0 +
        y[IDX_DI]*2.0 + y[IDX_DII]*2.0 + y[IDX_DM]*2.0 + y[IDX_D2OI]*20.0 +
        y[IDX_D2OII]*20.0 + y[IDX_D3OII]*22.0 + y[IDX_DCNI]*28.0 +
        y[IDX_DCNII]*28.0 + y[IDX_DCOI]*30.0 + y[IDX_DCOII]*30.0 +
        y[IDX_DNCI]*28.0 + y[IDX_DNCII]*28.0 + y[IDX_DNOI]*32.0 +
        y[IDX_DNOII]*32.0 + y[IDX_DOCII]*30.0 + y[IDX_GRAINM]*0.0 +
        y[IDX_GRAIN0I]*0.0 + y[IDX_HI]*1.0 + y[IDX_HII]*1.0 + y[IDX_HM]*1.0 +
        y[IDX_H2DOII]*20.0 + y[IDX_H2OI]*18.0 + y[IDX_H2OII]*18.0 +
        y[IDX_H3OII]*19.0 + y[IDX_HCNI]*27.0 + y[IDX_HCNII]*27.0 +
        y[IDX_HCOI]*29.0 + y[IDX_HCOII]*29.0 + y[IDX_HDI]*3.0 + y[IDX_HDII]*3.0
        + y[IDX_HD2OII]*21.0 + y[IDX_HDOI]*19.0 + y[IDX_HDOII]*19.0 +
        y[IDX_HNCI]*27.0 + y[IDX_HNCII]*27.0 + y[IDX_HNOI]*31.0 +
        y[IDX_HNOII]*31.0 + y[IDX_HOCII]*29.0 + y[IDX_HeI]*4.0 + y[IDX_HeII]*4.0
        + y[IDX_HeDII]*6.0 + y[IDX_HeHII]*5.0 + y[IDX_NI]*14.0 + y[IDX_NII]*14.0
        + y[IDX_N2I]*28.0 + y[IDX_N2II]*28.0 + y[IDX_N2DII]*30.0 +
        y[IDX_N2HII]*29.0 + y[IDX_N2OI]*44.0 + y[IDX_NCOII]*42.0 +
        y[IDX_NDI]*16.0 + y[IDX_NDII]*16.0 + y[IDX_ND2I]*18.0 +
        y[IDX_ND2II]*18.0 + y[IDX_NHI]*15.0 + y[IDX_NHII]*15.0 +
        y[IDX_NH2I]*16.0 + y[IDX_NH2II]*16.0 + y[IDX_NHDI]*17.0 +
        y[IDX_NHDII]*17.0 + y[IDX_NOI]*30.0 + y[IDX_NOII]*30.0 +
        y[IDX_NO2I]*46.0 + y[IDX_NO2II]*46.0 + y[IDX_OI]*16.0 + y[IDX_OII]*16.0
        + y[IDX_OM]*16.0 + y[IDX_O2I]*32.0 + y[IDX_O2II]*32.0 + y[IDX_O2DI]*34.0
        + y[IDX_O2DII]*34.0 + y[IDX_O2HI]*33.0 + y[IDX_O2HII]*33.0 +
        y[IDX_OCNI]*42.0 + y[IDX_ODI]*18.0 + y[IDX_ODII]*18.0 + y[IDX_ODM]*18.0
        + y[IDX_OHI]*17.0 + y[IDX_OHII]*17.0 + y[IDX_OHM]*17.0 + y[IDX_eM]*0.0 +
        y[IDX_mD3II]*6.0 + y[IDX_oD2I]*4.0 + y[IDX_oD2II]*4.0 +
        y[IDX_oD2HII]*5.0 + y[IDX_oD3II]*6.0 + y[IDX_oH2I]*2.0 +
        y[IDX_oH2II]*2.0 + y[IDX_oH2DII]*4.0 + y[IDX_oH3II]*3.0 +
        y[IDX_pD2I]*4.0 + y[IDX_pD2II]*4.0 + y[IDX_pD2HII]*5.0 +
        y[IDX_pD3II]*6.0 + y[IDX_pH2I]*2.0 + y[IDX_pH2II]*2.0 +
        y[IDX_pH2DII]*4.0 + y[IDX_pH3II]*3.0) / (y[IDX_CI] + y[IDX_CII] +
        y[IDX_CM] + y[IDX_C2I] + y[IDX_C2II] + y[IDX_C2DI] + y[IDX_C2DII] +
        y[IDX_C2HI] + y[IDX_C2HII] + y[IDX_C2NI] + y[IDX_C2NII] + y[IDX_C2OII] +
        y[IDX_C3I] + y[IDX_C3II] + y[IDX_CCOI] + y[IDX_CDI] + y[IDX_CDII] +
        y[IDX_CD2I] + y[IDX_CD2II] + y[IDX_CHI] + y[IDX_CHII] + y[IDX_CH2I] +
        y[IDX_CH2II] + y[IDX_CHDI] + y[IDX_CHDII] + y[IDX_CNI] + y[IDX_CNII] +
        y[IDX_CNM] + y[IDX_CNCII] + y[IDX_COI] + y[IDX_COII] + y[IDX_CO2I] +
        y[IDX_CO2II] + y[IDX_DI] + y[IDX_DII] + y[IDX_DM] + y[IDX_D2OI] +
        y[IDX_D2OII] + y[IDX_D3OII] + y[IDX_DCNI] + y[IDX_DCNII] + y[IDX_DCOI] +
        y[IDX_DCOII] + y[IDX_DNCI] + y[IDX_DNCII] + y[IDX_DNOI] + y[IDX_DNOII] +
        y[IDX_DOCII] + y[IDX_GRAINM] + y[IDX_GRAIN0I] + y[IDX_HI] + y[IDX_HII] +
        y[IDX_HM] + y[IDX_H2DOII] + y[IDX_H2OI] + y[IDX_H2OII] + y[IDX_H3OII] +
        y[IDX_HCNI] + y[IDX_HCNII] + y[IDX_HCOI] + y[IDX_HCOII] + y[IDX_HDI] +
        y[IDX_HDII] + y[IDX_HD2OII] + y[IDX_HDOI] + y[IDX_HDOII] + y[IDX_HNCI] +
        y[IDX_HNCII] + y[IDX_HNOI] + y[IDX_HNOII] + y[IDX_HOCII] + y[IDX_HeI] +
        y[IDX_HeII] + y[IDX_HeDII] + y[IDX_HeHII] + y[IDX_NI] + y[IDX_NII] +
        y[IDX_N2I] + y[IDX_N2II] + y[IDX_N2DII] + y[IDX_N2HII] + y[IDX_N2OI] +
        y[IDX_NCOII] + y[IDX_NDI] + y[IDX_NDII] + y[IDX_ND2I] + y[IDX_ND2II] +
        y[IDX_NHI] + y[IDX_NHII] + y[IDX_NH2I] + y[IDX_NH2II] + y[IDX_NHDI] +
        y[IDX_NHDII] + y[IDX_NOI] + y[IDX_NOII] + y[IDX_NO2I] + y[IDX_NO2II] +
        y[IDX_OI] + y[IDX_OII] + y[IDX_OM] + y[IDX_O2I] + y[IDX_O2II] +
        y[IDX_O2DI] + y[IDX_O2DII] + y[IDX_O2HI] + y[IDX_O2HII] + y[IDX_OCNI] +
        y[IDX_ODI] + y[IDX_ODII] + y[IDX_ODM] + y[IDX_OHI] + y[IDX_OHII] +
        y[IDX_OHM] + y[IDX_eM] + y[IDX_mD3II] + y[IDX_oD2I] + y[IDX_oD2II] +
        y[IDX_oD2HII] + y[IDX_oD3II] + y[IDX_oH2I] + y[IDX_oH2II] +
        y[IDX_oH2DII] + y[IDX_oH3II] + y[IDX_pD2I] + y[IDX_pD2II] +
        y[IDX_pD2HII] + y[IDX_pD3II] + y[IDX_pH2I] + y[IDX_pH2II] +
        y[IDX_pH2DII] + y[IDX_pH3II]);
}

__device__ __host__ double GetGamma(double *y) {
    return 5.0 / 3.0;
}

__device__ __host__ double GetNumDens(double *y) {
    double numdens = 0.0;

    for (int i = 0; i < NSPECIES; i++) numdens += y[i];
    return numdens;
}
// clang-format on

// clang-format off
__device__ double GetShieldingFactor(int specidx, double h2coldens, double spcoldens,
                          double tgas, int method) {
    // clang-format on
    double factor;
#ifdef IDX_H2I
    if (specidx == IDX_H2I) {
        factor = GetH2shielding(h2coldens, method);
    }
#endif
#ifdef IDX_COI
    if (specidx == IDX_COI) {
        factor = GetCOshielding(tgas, h2coldens, spcoldens, method);
    }
#endif
#ifdef IDX_N2I
    if (specidx == IDX_N2I) {
        factor = GetN2shielding(tgas, h2coldens, spcoldens, method);
    }
#endif

    return factor;
}

// clang-format off
__device__ double GetH2shielding(double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetH2shieldingInt(coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
__device__ double GetCOshielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetCOshieldingInt(tgas, h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// clang-format off
__device__ double GetN2shielding(double tgas, double h2col, double coldens, int method) {
    // clang-format on
    double shielding = -1.0;
    switch (method) {
        case 0:
            shielding = GetN2shieldingInt(tgas, h2col, coldens);
            break;
        default:
            break;
    }
    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetH2shieldingInt(double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetCOshieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on
    double shielding = -1.0;

    /* */

    return shielding;
}

// Interpolate/Extropolate from table (must be rendered in naunet constants)
// clang-format off
__device__ double GetN2shieldingInt(double tgas, double h2col, double coldens) {
    // clang-format on

    double shielding = -1.0;

    /* */

    return shielding;
}