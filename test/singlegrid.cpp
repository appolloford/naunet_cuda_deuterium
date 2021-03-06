#include <stdio.h>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

// Maximal steps, equal to the rows of `timeres.dat` - 1
#define NTIMESTEPS 10045

int main() {

    int nsteps   = 50; // The steps to evolve, must less than NTIMESTEPS

    double spy   = 86400.0 * 365.0;
    double pi    = 3.14159265;
    double rD    = 1.0e-5;
    double rhoD  = 3.0;
    double DtoGM = 7.09e-3;
    double amH   = 1.66043e-24;
    double nH    = 1e5;
    double OPRH2 = 0.1;

    NaunetData data;
    data.nH          = nH;
    data.Tgas        = 15.0;
    data.user_Av     = 30.0;
    data.user_crflux = 2.5e-17;
    data.user_GtoDN  = (4.e0 * pi * rhoD * rD * rD * rD) / (3.e0 * DtoGM * amH);

    Naunet naunet;
    naunet.Init();

#ifdef USE_CUDA
    naunet.Reset(1);
#endif

    double y[NEQUATIONS];
    for (int i = 0; i < NEQUATIONS; i++) {
        y[i] = 1.e-40;
    }
    y[IDX_pH2I]    = 1.0 / (1.0 + OPRH2) * 0.5 * nH;
    y[IDX_oH2I]    = OPRH2 / (1.0 + OPRH2) * 0.5 * nH;
    y[IDX_HDI]     = 1.5e-5 * nH;
    y[IDX_HeI]     = 1.0e-1 * nH;
    y[IDX_NI]      = 2.1e-6 * nH;
    y[IDX_OI]      = 1.8e-5 * nH;
    y[IDX_CI]      = 7.3e-6 * nH;
    y[IDX_GRAIN0I] = 1.3215e-12 * nH;

    double time[NTIMESTEPS];
    FILE *tfile = fopen("timeres.dat", "r");
    for (int i = 0; i < NTIMESTEPS; i++) {
        fscanf(tfile, "%lf\n", time + i);
    }
    fclose(tfile);

    FILE *fbin = fopen("evolution_singlegrid.bin", "w");
    FILE *ftxt = fopen("evolution_singlegrid.txt", "w");
    FILE *ttxt = fopen("time_singlegrid.txt", "w");
#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    FILE *rtxt = fopen("reactionrates.txt", "w");
    double rates[NREACTIONS];
#endif

    double dtyr = 1.0, tend = 1.e8;
    // for (time = 0.0; time < tend; time += dtyr)
    // {
    //     if (time < 1e5)
    //     {
    //         dtyr = fmax(9.0 * time, dtyr);
    //     }
    //     else
    //     {
    //         dtyr = 1e5;
    //     }
    for (int i = 0; i < nsteps; i++) {
#ifdef NAUNET_DEBUG
        EvalRates(rates, y, &data);
        for (int j = 0; j < NREACTIONS; j++) {
            fprintf(rtxt, "%13.7e ", rates[j]);
        }
        fprintf(rtxt, "\n");
#endif

        dtyr = time[i + 1] - time[i];

        fwrite(time + i, sizeof(double), 1, fbin);
        fwrite(y, sizeof(double), NEQUATIONS, fbin);

        fprintf(ftxt, "%13.7e ", time[i]);
        for (int j = 0; j < NEQUATIONS; j++) {
            fprintf(ftxt, "%13.7e ", y[j]);
        }
        fprintf(ftxt, "\n");

        Timer timer;
        timer.start();
        naunet.Solve(y, dtyr * spy, &data);
        timer.stop();
        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        // printf("Time = %13.7e yr, elapsed: %8.5e sec\n", time[i + 1],
        // duration);
    }

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);
#ifdef NAUNET_DEBUG
    fclose(rtxt);
#endif

    naunet.Finalize();

    return 0;
}
