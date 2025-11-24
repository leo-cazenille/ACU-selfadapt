#ifndef ACU_SELFADAPT_UTILS_H
#define ACU_SELFADAPT_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "pogobase.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define SCALE_0_255_TO_0_25(x)   (uint8_t)((x) * (25.0f / 255.0f) + 0.5f)
static inline double wrap_pi(double a){
    while(a> M_PI)a-=2.0*M_PI;
    while(a<-M_PI)a+=2.0*M_PI;
    return a; 
}
static inline double wrap_2pi(double a){
    // Map any angle to [0, 2π).
    a = fmod(a, 2.0*M_PI);
    if (a < 0.0) a += 2.0*M_PI;
    return a;
}
static inline int16_t rad_to_mrad(double a){ a=wrap_pi(a); long v=lround(a*1000.0); if(v>32767)v=32767; if(v<-32768)v=-32768; return (int16_t)v; }
static inline double  mrad_to_rad(int16_t m){ return ((double)m)/1000.0; }
static inline double  rand_uniform(double a, double b){ double u=(double)rand()/(double)RAND_MAX; return a + (b - a) * u; }
static inline double  clamp01(double x){ if (x<0.0) return 0.0; if (x>1.0) return 1.0; return x; }
static inline double  clamp01s(double x){ if (x<-1.0) return -1.0; if (x>1.0) return 1.0; return x; }


/* Box–Muller Gaussian (C11-safe) */
static inline double randn_box_muller(void){
    double u1 = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
    double u2 = ((double)rand()+1.0)/((double)RAND_MAX+2.0);
    return sqrt(-2.0*log(u1)) * cos(2.0*M_PI*u2);
}

/* --- Safe normalize/denormalize helpers --- */
static inline double norm01_safe(double x, double lo, double hi){
    double d = hi - lo;
    if (fabs(d) < 1e-12) return 0.0;          /* fixed value, map to 0 in [0,1] */
    return (x - lo) / d;
}
static inline double denorm01_safe(double u, double lo, double hi){
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    return lo + u * (hi - lo);
}


/* Q15 pack for push-sum */
static inline int16_t q15_pack_double(double x, double xmin, double xmax){
    if (x < xmin) x = xmin;
    if (x > xmax) x = xmax;
    double xn = (x - xmin) / (xmax - xmin);
    return (int16_t)lround((xn * 2.0 - 1.0) * 32767.0);
}
static inline double q15_unpack_double(int16_t q, double xmin, double xmax){
    double xn = ((double)q) / 32767.0; double t  = (xn + 1.0) * 0.5; return xmin + t * (xmax - xmin);
}
static inline uint16_t q15_pack_w(double w){ if (w < 0.0) w = 0.0; if (w > 2.0) w = 2.0; return (uint16_t)lround(w / 2.0 * 65535.0); }
static inline double   q15_unpack_w(uint16_t qw){ return ((double)qw) / 65535.0 * 2.0; }

#ifdef __cplusplus
}
#endif

#endif /* ACU_SELFADAPT_UTILS_H */

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker
