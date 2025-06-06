/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Drone_ode_dyn_disc_phi_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_house CASADI_PREFIX(house)
#define casadi_if_else CASADI_PREFIX(if_else)
#define casadi_qr CASADI_PREFIX(qr)
#define casadi_qr_colcomb CASADI_PREFIX(qr_colcomb)
#define casadi_qr_mv CASADI_PREFIX(qr_mv)
#define casadi_qr_singular CASADI_PREFIX(qr_singular)
#define casadi_qr_solve CASADI_PREFIX(qr_solve)
#define casadi_qr_trs CASADI_PREFIX(qr_trs)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_scal CASADI_PREFIX(scal)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_if_else(casadi_real c, casadi_real x, casadi_real y) { return c!=0 ? x : y;}

void casadi_scal(casadi_int n, casadi_real alpha, casadi_real* x) {
  casadi_int i;
  if (!x) return;
  for (i=0; i<n; ++i) *x++ *= alpha;
}

casadi_real casadi_house(casadi_real* v, casadi_real* beta, casadi_int nv) {
  casadi_int i;
  casadi_real v0, sigma, s, sigma_is_zero, v0_nonpos;
  v0 = v[0];
  sigma=0;
  for (i=1; i<nv; ++i) sigma += v[i]*v[i];
  s = sqrt(v0*v0 + sigma);
  sigma_is_zero = sigma==0;
  v0_nonpos = v0<=0;
  v[0] = casadi_if_else(sigma_is_zero, 1,
                 casadi_if_else(v0_nonpos, v0-s, -sigma/(v0+s)));
  *beta = casadi_if_else(sigma_is_zero, 2*v0_nonpos, -1/(s*v[0]));
  return s;
}
void casadi_qr(const casadi_int* sp_a, const casadi_real* nz_a, casadi_real* x,
               const casadi_int* sp_v, casadi_real* nz_v, const casadi_int* sp_r, casadi_real* nz_r, casadi_real* beta,
               const casadi_int* prinv, const casadi_int* pc) {
   casadi_int ncol, nrow, r, c, k, k1;
   casadi_real alpha;
   const casadi_int *a_colind, *a_row, *v_colind, *v_row, *r_colind, *r_row;
   ncol = sp_a[1];
   a_colind=sp_a+2; a_row=sp_a+2+ncol+1;
   nrow = sp_v[0];
   v_colind=sp_v+2; v_row=sp_v+2+ncol+1;
   r_colind=sp_r+2; r_row=sp_r+2+ncol+1;
   for (r=0; r<nrow; ++r) x[r] = 0;
   for (c=0; c<ncol; ++c) {
     for (k=a_colind[pc[c]]; k<a_colind[pc[c]+1]; ++k) x[prinv[a_row[k]]] = nz_a[k];
     for (k=r_colind[c]; k<r_colind[c+1] && (r=r_row[k])<c; ++k) {
       alpha = 0;
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) alpha += nz_v[k1]*x[v_row[k1]];
       alpha *= beta[r];
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) x[v_row[k1]] -= alpha*nz_v[k1];
       *nz_r++ = x[r];
       x[r] = 0;
     }
     for (k=v_colind[c]; k<v_colind[c+1]; ++k) {
       nz_v[k] = x[v_row[k]];
       x[v_row[k]] = 0;
     }
     *nz_r++ = casadi_house(nz_v + v_colind[c], beta + c, v_colind[c+1] - v_colind[c]);
   }
 }
void casadi_qr_mv(const casadi_int* sp_v, const casadi_real* v, const casadi_real* beta, casadi_real* x,
                  casadi_int tr) {
  casadi_int ncol, c, c1, k;
  casadi_real alpha;
  const casadi_int *colind, *row;
  ncol=sp_v[1];
  colind=sp_v+2; row=sp_v+2+ncol+1;
  for (c1=0; c1<ncol; ++c1) {
    c = tr ? c1 : ncol-1-c1;
    alpha=0;
    for (k=colind[c]; k<colind[c+1]; ++k) alpha += v[k]*x[row[k]];
    alpha *= beta[c];
    for (k=colind[c]; k<colind[c+1]; ++k) x[row[k]] -= alpha*v[k];
  }
}
void casadi_qr_trs(const casadi_int* sp_r, const casadi_real* nz_r, casadi_real* x, casadi_int tr) {
  casadi_int ncol, r, c, k;
  const casadi_int *colind, *row;
  ncol=sp_r[1];
  colind=sp_r+2; row=sp_r+2+ncol+1;
  if (tr) {
    for (c=0; c<ncol; ++c) {
      for (k=colind[c]; k<colind[c+1]; ++k) {
        r = row[k];
        if (r==c) {
          x[c] /= nz_r[k];
        } else {
          x[c] -= nz_r[k]*x[r];
        }
      }
    }
  } else {
    for (c=ncol-1; c>=0; --c) {
      for (k=colind[c+1]-1; k>=colind[c]; --k) {
        r=row[k];
        if (r==c) {
          x[r] /= nz_r[k];
        } else {
          x[r] -= nz_r[k]*x[c];
        }
      }
    }
  }
}
void casadi_qr_solve(casadi_real* x, casadi_int nrhs, casadi_int tr,
                     const casadi_int* sp_v, const casadi_real* v, const casadi_int* sp_r, const casadi_real* r,
                     const casadi_real* beta, const casadi_int* prinv, const casadi_int* pc, casadi_real* w) {
  casadi_int k, c, nrow_ext, ncol;
  nrow_ext = sp_v[0]; ncol = sp_v[1];
  for (k=0; k<nrhs; ++k) {
    if (tr) {
      for (c=0; c<ncol; ++c) w[c] = x[pc[c]];
      casadi_qr_trs(sp_r, r, w, 1);
      casadi_qr_mv(sp_v, v, beta, w, 0);
      for (c=0; c<ncol; ++c) x[c] = w[prinv[c]];
    } else {
      for (c=0; c<nrow_ext; ++c) w[c] = 0;
      for (c=0; c<ncol; ++c) w[prinv[c]] = x[c];
      casadi_qr_mv(sp_v, v, beta, w, 1);
      casadi_qr_trs(sp_r, r, w, 0);
      for (c=0; c<ncol; ++c) x[pc[c]] = w[c];
    }
    x += ncol;
  }
}
casadi_int casadi_qr_singular(casadi_real* rmin, casadi_int* irmin, const casadi_real* nz_r,
                             const casadi_int* sp_r, const casadi_int* pc, casadi_real eps) {
  casadi_real rd, rd_min;
  casadi_int ncol, c, nullity;
  const casadi_int* r_colind;
  nullity = 0;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  for (c=0; c<ncol; ++c) {
    rd = fabs(nz_r[r_colind[c+1]-1]);
    if (rd<eps) nullity++;
    if (c==0 || rd < rd_min) {
      rd_min = rd;
      if (rmin) *rmin = rd;
      if (irmin) *irmin = pc[c];
    }
  }
  return nullity;
}
void casadi_qr_colcomb(casadi_real* v, const casadi_real* nz_r, const casadi_int* sp_r,
                       const casadi_int* pc, casadi_real eps, casadi_int ind) {
  casadi_int ncol, r, c, k;
  const casadi_int *r_colind, *r_row;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  r_row = r_colind + ncol + 1;
  for (c=0; c<ncol; ++c) {
    if (fabs(nz_r[r_colind[c+1]-1])<eps && 0==ind--) {
      ind = c;
      break;
    }
  }
  casadi_clear(v, ncol);
  v[pc[ind]] = 1.;
  for (k=r_colind[ind]; k<r_colind[ind+1]-1; ++k) {
    v[pc[r_row[k]]] = -nz_r[k];
  }
  for (c=ind-1; c>=0; --c) {
    for (k=r_colind[c+1]-1; k>=r_colind[c]; --k) {
      r=r_row[k];
      if (r==c) {
        if (fabs(nz_r[k])<eps) {
          v[pc[r]] = 0;
        } else {
          v[pc[r]] /= nz_r[k];
        }
      } else {
        v[pc[r]] -= nz_r[k]*v[pc[c]];
      }
    }
  }
  casadi_scal(ncol, 1./sqrt(casadi_dot(ncol, v, v)), v);
}

static const casadi_int casadi_s0[4] = {0, 1, 2, 3};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[17] = {4, 4, 0, 4, 7, 9, 10, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
static const casadi_int casadi_s3[17] = {4, 4, 0, 1, 3, 6, 10, 0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
static const casadi_int casadi_s4[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s5[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s6[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[16] = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};

/* Drone_ode_dyn_disc_phi_fun:(i0[11],i1[4],i2[15])->(o0[11]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+11, *w1=w+22, *w2=w+43, *w3=w+52, *w4=w+61, *w5=w+70, w6, w7, w8, w9, *w10=w+85, *w11=w+89, *w12=w+98, *w13=w+101, *w14=w+134, *w15=w+167, w16, w17, *w18=w+181, *w19=w+185, *w20=w+189, *w21=w+193, *w22=w+209, *w23=w+225, *w24=w+269, *w25=w+313, *w26=w+341, *w27=w+357, *w28=w+373, *w29=w+417, *w30=w+538;
  /* #0: @0 = zeros(11x1) */
  casadi_clear(w0, 11);
  /* #1: @1 = zeros(3x7) */
  casadi_clear(w1, 21);
  /* #2: @2 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w2);
  /* #3: @3 = zeros(3x3) */
  casadi_clear(w3, 9);
  /* #4: @4 = zeros(3x3) */
  casadi_clear(w4, 9);
  /* #5: @5 = input[0][0] */
  casadi_copy(arg[0], 11, w5);
  /* #6: @6 = @5[3] */
  for (rr=(&w6), ss=w5+3; ss!=w5+4; ss+=1) *rr++ = *ss;
  /* #7: @7 = @5[4] */
  for (rr=(&w7), ss=w5+4; ss!=w5+5; ss+=1) *rr++ = *ss;
  /* #8: @8 = @5[5] */
  for (rr=(&w8), ss=w5+5; ss!=w5+6; ss+=1) *rr++ = *ss;
  /* #9: @9 = @5[6] */
  for (rr=(&w9), ss=w5+6; ss!=w5+7; ss+=1) *rr++ = *ss;
  /* #10: @10 = vertcat(@6, @7, @8, @9) */
  rr=w10;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #11: @6 = ||@10||_F */
  w6 = sqrt(casadi_dot(4, w10, w10));
  /* #12: @10 = (@10/@6) */
  for (i=0, rr=w10; i<4; ++i) (*rr++) /= w6;
  /* #13: @6 = @10[3] */
  for (rr=(&w6), ss=w10+3; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #14: @6 = (-@6) */
  w6 = (- w6 );
  /* #15: (@4[3] = @6) */
  for (rr=w4+3, ss=(&w6); rr!=w4+4; rr+=1) *rr = *ss++;
  /* #16: @6 = @10[2] */
  for (rr=(&w6), ss=w10+2; ss!=w10+3; ss+=1) *rr++ = *ss;
  /* #17: (@4[6] = @6) */
  for (rr=w4+6, ss=(&w6); rr!=w4+7; rr+=1) *rr = *ss++;
  /* #18: @6 = @10[1] */
  for (rr=(&w6), ss=w10+1; ss!=w10+2; ss+=1) *rr++ = *ss;
  /* #19: @6 = (-@6) */
  w6 = (- w6 );
  /* #20: (@4[7] = @6) */
  for (rr=w4+7, ss=(&w6); rr!=w4+8; rr+=1) *rr = *ss++;
  /* #21: @6 = @10[3] */
  for (rr=(&w6), ss=w10+3; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #22: (@4[1] = @6) */
  for (rr=w4+1, ss=(&w6); rr!=w4+2; rr+=1) *rr = *ss++;
  /* #23: @6 = @10[2] */
  for (rr=(&w6), ss=w10+2; ss!=w10+3; ss+=1) *rr++ = *ss;
  /* #24: @6 = (-@6) */
  w6 = (- w6 );
  /* #25: (@4[2] = @6) */
  for (rr=w4+2, ss=(&w6); rr!=w4+3; rr+=1) *rr = *ss++;
  /* #26: @6 = @10[1] */
  for (rr=(&w6), ss=w10+1; ss!=w10+2; ss+=1) *rr++ = *ss;
  /* #27: (@4[5] = @6) */
  for (rr=w4+5, ss=(&w6); rr!=w4+6; rr+=1) *rr = *ss++;
  /* #28: @11 = (2.*@4) */
  for (i=0, rr=w11, cs=w4; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #29: @3 = mac(@11,@4,@3) */
  for (i=0, rr=w3; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w11+j, tt=w4+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #30: @2 = (@2+@3) */
  for (i=0, rr=w2, cs=w3; i<9; ++i) (*rr++) += (*cs++);
  /* #31: @6 = @10[0] */
  for (rr=(&w6), ss=w10+0; ss!=w10+1; ss+=1) *rr++ = *ss;
  /* #32: @6 = (2.*@6) */
  w6 = (2.* w6 );
  /* #33: @4 = (@6*@4) */
  for (i=0, rr=w4, cs=w4; i<9; ++i) (*rr++)  = (w6*(*cs++));
  /* #34: @2 = (@2+@4) */
  for (i=0, rr=w2, cs=w4; i<9; ++i) (*rr++) += (*cs++);
  /* #35: @12 = zeros(3x1) */
  casadi_clear(w12, 3);
  /* #36: @13 = horzcat(@1, @2, @12) */
  rr=w13;
  for (i=0, cs=w1; i<21; ++i) *rr++ = *cs++;
  for (i=0, cs=w2; i<9; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<3; ++i) *rr++ = *cs++;
  /* #37: @14 = @13' */
  for (i=0, rr=w14, cs=w13; i<11; ++i) for (j=0; j<3; ++j) rr[i+j*11] = *cs++;
  /* #38: @15 = zeros(4x3) */
  casadi_clear(w15, 12);
  /* #39: @6 = 0.5 */
  w6 = 5.0000000000000000e-01;
  /* #40: @7 = 0 */
  w7 = 0.;
  /* #41: @8 = 0 */
  w8 = 0.;
  /* #42: @9 = 0 */
  w9 = 0.;
  /* #43: @16 = @5[10] */
  for (rr=(&w16), ss=w5+10; ss!=w5+11; ss+=1) *rr++ = *ss;
  /* #44: @17 = (-@16) */
  w17 = (- w16 );
  /* #45: @10 = horzcat(@7, @8, @9, @17) */
  rr=w10;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w17;
  /* #46: @10 = @10' */
  /* #47: @7 = 0 */
  w7 = 0.;
  /* #48: @8 = 0 */
  w8 = 0.;
  /* #49: @9 = 0 */
  w9 = 0.;
  /* #50: @18 = horzcat(@7, @8, @16, @9) */
  rr=w18;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w16;
  *rr++ = w9;
  /* #51: @18 = @18' */
  /* #52: @7 = 0 */
  w7 = 0.;
  /* #53: @8 = (-@16) */
  w8 = (- w16 );
  /* #54: @9 = 0 */
  w9 = 0.;
  /* #55: @17 = 0 */
  w17 = 0.;
  /* #56: @19 = horzcat(@7, @8, @9, @17) */
  rr=w19;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w17;
  /* #57: @19 = @19' */
  /* #58: @7 = 0 */
  w7 = 0.;
  /* #59: @8 = 0 */
  w8 = 0.;
  /* #60: @9 = 0 */
  w9 = 0.;
  /* #61: @20 = horzcat(@16, @7, @8, @9) */
  rr=w20;
  *rr++ = w16;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  /* #62: @20 = @20' */
  /* #63: @21 = horzcat(@10, @18, @19, @20) */
  rr=w21;
  for (i=0, cs=w10; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w18; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w19; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<4; ++i) *rr++ = *cs++;
  /* #64: @22 = @21' */
  for (i=0, rr=w22, cs=w21; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #65: @22 = (@6*@22) */
  for (i=0, rr=w22, cs=w22; i<16; ++i) (*rr++)  = (w6*(*cs++));
  /* #66: @21 = zeros(4x4) */
  casadi_clear(w21, 16);
  /* #67: @23 = horzcat(@15, @22, @21) */
  rr=w23;
  for (i=0, cs=w15; i<12; ++i) *rr++ = *cs++;
  for (i=0, cs=w22; i<16; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<16; ++i) *rr++ = *cs++;
  /* #68: @24 = @23' */
  for (i=0, rr=w24, cs=w23; i<11; ++i) for (j=0; j<4; ++j) rr[i+j*11] = *cs++;
  /* #69: @25 = zeros(4x7) */
  casadi_clear(w25, 28);
  /* #70: @22 = zeros(4x4) */
  casadi_clear(w22, 16);
  /* #71: @21 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c1, 16, w21);
  /* #72: @26 = zeros(4x4) */
  casadi_clear(w26, 16);
  /* #73: @6 = 0.6756 */
  w6 = 6.7559999999999998e-01;
  /* #74: (@26[0] = @6) */
  for (rr=w26+0, ss=(&w6); rr!=w26+1; rr+=1) *rr = *ss++;
  /* #75: @6 = 0 */
  w6 = 0.;
  /* #76: (@26[4] = @6) */
  for (rr=w26+4, ss=(&w6); rr!=w26+5; rr+=1) *rr = *ss++;
  /* #77: @6 = 0 */
  w6 = 0.;
  /* #78: (@26[8] = @6) */
  for (rr=w26+8, ss=(&w6); rr!=w26+9; rr+=1) *rr = *ss++;
  /* #79: @6 = 0 */
  w6 = 0.;
  /* #80: (@26[12] = @6) */
  for (rr=w26+12, ss=(&w6); rr!=w26+13; rr+=1) *rr = *ss++;
  /* #81: @6 = 0 */
  w6 = 0.;
  /* #82: (@26[1] = @6) */
  for (rr=w26+1, ss=(&w6); rr!=w26+2; rr+=1) *rr = *ss++;
  /* #83: @6 = 0.6344 */
  w6 = 6.3439999999999996e-01;
  /* #84: (@26[5] = @6) */
  for (rr=w26+5, ss=(&w6); rr!=w26+6; rr+=1) *rr = *ss++;
  /* #85: @6 = 0 */
  w6 = 0.;
  /* #86: (@26[9] = @6) */
  for (rr=w26+9, ss=(&w6); rr!=w26+10; rr+=1) *rr = *ss++;
  /* #87: @6 = 0 */
  w6 = 0.;
  /* #88: (@26[13] = @6) */
  for (rr=w26+13, ss=(&w6); rr!=w26+14; rr+=1) *rr = *ss++;
  /* #89: @6 = 0 */
  w6 = 0.;
  /* #90: (@26[2] = @6) */
  for (rr=w26+2, ss=(&w6); rr!=w26+3; rr+=1) *rr = *ss++;
  /* #91: @6 = 0 */
  w6 = 0.;
  /* #92: (@26[6] = @6) */
  for (rr=w26+6, ss=(&w6); rr!=w26+7; rr+=1) *rr = *ss++;
  /* #93: @6 = 0.408 */
  w6 = 4.0799999999999997e-01;
  /* #94: (@26[10] = @6) */
  for (rr=w26+10, ss=(&w6); rr!=w26+11; rr+=1) *rr = *ss++;
  /* #95: @6 = 0 */
  w6 = 0.;
  /* #96: (@26[14] = @6) */
  for (rr=w26+14, ss=(&w6); rr!=w26+15; rr+=1) *rr = *ss++;
  /* #97: @6 = 0 */
  w6 = 0.;
  /* #98: (@26[3] = @6) */
  for (rr=w26+3, ss=(&w6); rr!=w26+4; rr+=1) *rr = *ss++;
  /* #99: @6 = 0 */
  w6 = 0.;
  /* #100: (@26[7] = @6) */
  for (rr=w26+7, ss=(&w6); rr!=w26+8; rr+=1) *rr = *ss++;
  /* #101: @6 = 0 */
  w6 = 0.;
  /* #102: (@26[11] = @6) */
  for (rr=w26+11, ss=(&w6); rr!=w26+12; rr+=1) *rr = *ss++;
  /* #103: @6 = 0.2953 */
  w6 = 2.9530000000000001e-01;
  /* #104: (@26[15] = @6) */
  for (rr=w26+15, ss=(&w6); rr!=w26+16; rr+=1) *rr = *ss++;
  /* #105: @21 = (@26\@21) */
  rr = w21;
  ss = w26;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #106: @27 = zeros(4x4) */
  casadi_clear(w27, 16);
  /* #107: @6 = 0.5941 */
  w6 = 5.9409999999999996e-01;
  /* #108: (@27[0] = @6) */
  for (rr=w27+0, ss=(&w6); rr!=w27+1; rr+=1) *rr = *ss++;
  /* #109: @6 = -0.8109 */
  w6 = -8.1089999999999995e-01;
  /* #110: @6 = (@6*@16) */
  w6 *= w16;
  /* #111: (@27[4] = @6) */
  for (rr=w27+4, ss=(&w6); rr!=w27+5; rr+=1) *rr = *ss++;
  /* #112: @6 = 0 */
  w6 = 0.;
  /* #113: (@27[8] = @6) */
  for (rr=w27+8, ss=(&w6); rr!=w27+9; rr+=1) *rr = *ss++;
  /* #114: @6 = 0 */
  w6 = 0.;
  /* #115: (@27[12] = @6) */
  for (rr=w27+12, ss=(&w6); rr!=w27+13; rr+=1) *rr = *ss++;
  /* #116: @6 = 0.3984 */
  w6 = 3.9839999999999998e-01;
  /* #117: @6 = (@6*@16) */
  w6 *= w16;
  /* #118: (@27[1] = @6) */
  for (rr=w27+1, ss=(&w6); rr!=w27+2; rr+=1) *rr = *ss++;
  /* #119: @6 = 0.704 */
  w6 = 7.0399999999999996e-01;
  /* #120: (@27[5] = @6) */
  for (rr=w27+5, ss=(&w6); rr!=w27+6; rr+=1) *rr = *ss++;
  /* #121: @6 = 0 */
  w6 = 0.;
  /* #122: (@27[9] = @6) */
  for (rr=w27+9, ss=(&w6); rr!=w27+10; rr+=1) *rr = *ss++;
  /* #123: @6 = 0 */
  w6 = 0.;
  /* #124: (@27[13] = @6) */
  for (rr=w27+13, ss=(&w6); rr!=w27+14; rr+=1) *rr = *ss++;
  /* #125: @6 = 0 */
  w6 = 0.;
  /* #126: (@27[2] = @6) */
  for (rr=w27+2, ss=(&w6); rr!=w27+3; rr+=1) *rr = *ss++;
  /* #127: @6 = 0 */
  w6 = 0.;
  /* #128: (@27[6] = @6) */
  for (rr=w27+6, ss=(&w6); rr!=w27+7; rr+=1) *rr = *ss++;
  /* #129: @6 = 0.9365 */
  w6 = 9.3650000000000000e-01;
  /* #130: (@27[10] = @6) */
  for (rr=w27+10, ss=(&w6); rr!=w27+11; rr+=1) *rr = *ss++;
  /* #131: @6 = 0 */
  w6 = 0.;
  /* #132: (@27[14] = @6) */
  for (rr=w27+14, ss=(&w6); rr!=w27+15; rr+=1) *rr = *ss++;
  /* #133: @6 = 0 */
  w6 = 0.;
  /* #134: (@27[3] = @6) */
  for (rr=w27+3, ss=(&w6); rr!=w27+4; rr+=1) *rr = *ss++;
  /* #135: @6 = 0 */
  w6 = 0.;
  /* #136: (@27[7] = @6) */
  for (rr=w27+7, ss=(&w6); rr!=w27+8; rr+=1) *rr = *ss++;
  /* #137: @6 = 0 */
  w6 = 0.;
  /* #138: (@27[11] = @6) */
  for (rr=w27+11, ss=(&w6); rr!=w27+12; rr+=1) *rr = *ss++;
  /* #139: @6 = 0.9752 */
  w6 = 9.7519999999999996e-01;
  /* #140: (@27[15] = @6) */
  for (rr=w27+15, ss=(&w6); rr!=w27+16; rr+=1) *rr = *ss++;
  /* #141: @22 = mac(@21,@27,@22) */
  for (i=0, rr=w22; i<4; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w21+j, tt=w27+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #142: @22 = (-@22) */
  for (i=0, rr=w22, cs=w22; i<16; ++i) *rr++ = (- *cs++ );
  /* #143: @23 = horzcat(@25, @22) */
  rr=w23;
  for (i=0, cs=w25; i<28; ++i) *rr++ = *cs++;
  for (i=0, cs=w22; i<16; ++i) *rr++ = *cs++;
  /* #144: @28 = @23' */
  for (i=0, rr=w28, cs=w23; i<11; ++i) for (j=0; j<4; ++j) rr[i+j*11] = *cs++;
  /* #145: @29 = horzcat(@14, @24, @28) */
  rr=w29;
  for (i=0, cs=w14; i<33; ++i) *rr++ = *cs++;
  for (i=0, cs=w24; i<44; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<44; ++i) *rr++ = *cs++;
  /* #146: @30 = @29' */
  for (i=0, rr=w30, cs=w29; i<11; ++i) for (j=0; j<11; ++j) rr[i+j*11] = *cs++;
  /* #147: @0 = mac(@30,@5,@0) */
  for (i=0, rr=w0; i<1; ++i) for (j=0; j<11; ++j, ++rr) for (k=0, ss=w30+j, tt=w5+i*11; k<11; ++k) *rr += ss[k*11]**tt++;
  /* #148: @5 = zeros(11x1) */
  casadi_clear(w5, 11);
  /* #149: @25 = zeros(4x7) */
  casadi_clear(w25, 28);
  /* #150: @22 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c1, 16, w22);
  /* #151: @22 = (@26\@22) */
  rr = w22;
  ss = w26;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[10], r[10], beta[4], w[8];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 4, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #152: @26 = @22' */
  for (i=0, rr=w26, cs=w22; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #153: @24 = horzcat(@25, @26) */
  rr=w24;
  for (i=0, cs=w25; i<28; ++i) *rr++ = *cs++;
  for (i=0, cs=w26; i<16; ++i) *rr++ = *cs++;
  /* #154: @28 = @24' */
  for (i=0, rr=w28, cs=w24; i<11; ++i) for (j=0; j<4; ++j) rr[i+j*11] = *cs++;
  /* #155: @10 = input[1][0] */
  casadi_copy(arg[1], 4, w10);
  /* #156: @5 = mac(@28,@10,@5) */
  for (i=0, rr=w5; i<1; ++i) for (j=0; j<11; ++j, ++rr) for (k=0, ss=w28+j, tt=w10+i*4; k<4; ++k) *rr += ss[k*11]**tt++;
  /* #157: @0 = (@0+@5) */
  for (i=0, rr=w0, cs=w5; i<11; ++i) (*rr++) += (*cs++);
  /* #158: output[0][0] = @0 */
  casadi_copy(w0, 11, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_dyn_disc_phi_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_dyn_disc_phi_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_dyn_disc_phi_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_dyn_disc_phi_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_dyn_disc_phi_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_dyn_disc_phi_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_dyn_disc_phi_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_dyn_disc_phi_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_dyn_disc_phi_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_dyn_disc_phi_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_dyn_disc_phi_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_dyn_disc_phi_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_dyn_disc_phi_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_dyn_disc_phi_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_dyn_disc_phi_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_dyn_disc_phi_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 659;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
