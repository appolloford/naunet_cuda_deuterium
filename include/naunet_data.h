#ifndef __NAUNET_DATA_H__
#define __NAUNET_DATA_H__

// Struct for holding the nessesary additional variables for the problem.
struct NaunetData {
    // clang-format off
    double nH;
    double Tgas;
    double user_crflux;
    double user_Av;
    double user_GtoDN;
    
    // clang-format on
};

#endif