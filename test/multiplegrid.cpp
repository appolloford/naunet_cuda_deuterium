#include <stdio.h>

#include "naunet.h"
#include "naunet_data.h"
#include "naunet_macros.h"
#include "naunet_ode.h"
#include "naunet_timer.h"

// Maximal steps, equal to the rows of `timeres.dat` - 1
#define NTIMESTEPS 10045

int main() {
    int nsystem      = 1024;
    int nsteps       = 50;
    double spy       = 86400.0 * 365.0;
    double pi        = 3.14159265;
    double rD        = 1.0e-5;
    double rhoD      = 3.0;
    double DtoGM     = 7.09e-3;
    double amH       = 1.66043e-24;
    double nH        = 1e5;
    double OPRH2     = 0.1;

    NaunetData *data = new NaunetData[nsystem];
    for (int isys = 0; isys < nsystem; isys++) {
        data[isys].nH          = nH;
        data[isys].Tgas        = 15.0;
        data[isys].user_Av     = 30.0;
        data[isys].user_crflux = 2.5e-17;
        data[isys].user_GtoDN =
            (4.e0 * pi * rhoD * rD * rD * rD) / (3.e0 * DtoGM * amH);
    }

    Naunet naunet;
    naunet.Init();

    naunet.Reset(nsystem);

    double *y = new double[nsystem * NEQUATIONS];
    for (int isys = 0; isys < nsystem; isys++) {
        for (int i = 0; i < NEQUATIONS; i++) {
            y[isys * NEQUATIONS + i] = 1.e-40;
        }
        y[isys * NEQUATIONS + IDX_pH2I]    = 1.0 / (1.0 + OPRH2) * 0.5 * nH;
        y[isys * NEQUATIONS + IDX_oH2I]    = OPRH2 / (1.0 + OPRH2) * 0.5 * nH;
        y[isys * NEQUATIONS + IDX_HDI]     = 1.5e-5 * nH;
        y[isys * NEQUATIONS + IDX_HeI]     = 1.0e-1 * nH;
        y[isys * NEQUATIONS + IDX_NI]      = 2.1e-6 * nH;
        y[isys * NEQUATIONS + IDX_OI]      = 1.8e-5 * nH;
        y[isys * NEQUATIONS + IDX_CI]      = 7.3e-6 * nH;
        y[isys * NEQUATIONS + IDX_GRAIN0I] = 1.3215e-12 * nH;
    }

    double time[NTIMESTEPS];
    FILE *tfile = fopen("timeres.dat", "r");
    for (int i = 0; i < NTIMESTEPS; i++) {
        fscanf(tfile, "%lf\n", time + i);
    }
    fclose(tfile);

    FILE *fbin = fopen("evolution_multiplegrid.bin", "w");
    FILE *ftxt = fopen("evolution_multiplegrid.txt", "w");
    FILE *ttxt = fopen("time_parallel.txt", "w");

#ifdef NAUNET_DEBUG
    printf("Initialization is done. Start to evolve.\n");
    // FILE *rtxt = fopen("reactionrates.txt", "w");
    // double rates[NREACTIONS];
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
        // EvalRates only receive one system as input, disabled in parallel test
        // EvalRates(rates, y, data);
        // for (int j = 0; j < NREACTIONS; j++) {
        //     fprintf(rtxt, "%13.7e ", rates[j]);
        // }
        // fprintf(rtxt, "\n");
#endif

        dtyr = time[i + 1] - time[i];

        for (int isys = 0; isys < nsystem; isys++) {
            fwrite((double *)&isys, sizeof(double), 1, fbin);
            fwrite(time + i, sizeof(double), 1, fbin);
            fwrite(&y[isys * NEQUATIONS], sizeof(double), NEQUATIONS, fbin);

            fprintf(ftxt, "%13.7e ", (double)isys);
            fprintf(ftxt, "%13.7e ", time[i]);
            for (int j = 0; j < NEQUATIONS; j++) {
                fprintf(ftxt, "%13.7e ", y[isys * NEQUATIONS + j]);
            }
            fprintf(ftxt, "\n");
        }

        Timer timer;
        timer.start();
        naunet.Solve(y, dtyr * spy, data);
        timer.stop();
        // float duration = (float)timer.elapsed() / 1e6;
        double duration = timer.elapsed();
        fprintf(ttxt, "%8.5e \n", duration);
        printf("Time = %13.7e yr, elapsed: %8.5e sec\n", time[i + 1],
        duration);
    }

    fclose(fbin);
    fclose(ftxt);
    fclose(ttxt);

#ifdef NAUNET_DEBUG
    // fclose(rtxt);
#endif

    naunet.Finalize();

    delete[] data;
    delete[] y;

    return 0;
}
