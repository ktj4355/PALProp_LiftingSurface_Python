# vortex.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs, pi, pow, exp
cimport cython

#----------------------------------------------------------
# 기본 벡터 연산: 3차원 벡터의 norm, cross 등
#----------------------------------------------------------
@cython.inline
cdef inline double norm3(double a, double b, double c):
    return sqrt(a*a + b*b + c*c)

@cython.inline
cdef inline void cross3(double a0, double a1, double a2,
                        double b0, double b1, double b2,
                        double* out0, double* out1, double* out2):
    out0[0] = a1*b2 - a2*b1
    out1[0] = a2*b0 - a0*b2
    out2[0] = a0*b1 - a1*b0

#----------------------------------------------------------
# Vortex_Vatistas 함수 (핵심 와류 유도 계산)
# 입력: A, B, ColocationPoint 는 길이 3인 1차원 메모리뷰 (double[::1])
#----------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=1] Vortex_Vatistas(double[:] A,
                                                 double[:] B,
                                                 double[:] ColocationPoint,
                                                 double vortexStrength,
                                                 double rc,
                                                 double n):
    cdef double r1_0, r1_1, r1_2
    cdef double r2_0, r2_1, r2_2
    cdef double dL, r, cross0, cross1, cross2, cross_norm
    cdef double r1s, r2s, BA0, BA1, BA2, dot1, dot2, cosA, cosB, r_norm, vortexFactor
    cdef np.ndarray[double, ndim=1] Vout = np.zeros(3, dtype=np.float64)

    # r1 = ColocationPoint - A
    r1_0 = ColocationPoint[0] - A[0]
    r1_1 = ColocationPoint[1] - A[1]
    r1_2 = ColocationPoint[2] - A[2]
    # r2 = ColocationPoint - B
    r2_0 = ColocationPoint[0] - B[0]
    r2_1 = ColocationPoint[1] - B[1]
    r2_2 = ColocationPoint[2] - B[2]

    # dL = norm(A - B)
    dL = sqrt((A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]))
    if dL < 1e-12:
        return Vout
    # cross = r1 x r2
    cross3(r1_0, r1_1, r1_2, r2_0, r2_1, r2_2, &cross0, &cross1, &cross2)
    cross_norm = sqrt(cross0*cross0 + cross1*cross1 + cross2*cross2)
    r = cross_norm / dL
    if r < 1e-6 or vortexStrength < 1e-5:
        return Vout
    r1s = sqrt(r1_0*r1_0 + r1_1*r1_1 + r1_2*r1_2)
    r2s = sqrt(r2_0*r2_0 + r2_1*r2_1 + r2_2*r2_2)
    if r1s < 1e-12 or r2s < 1e-12:
        return Vout
    # BA = B - A
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    BA2 = B[2] - A[2]
    dot1 = r1_0*BA0 + r1_1*BA1 + r1_2*BA2
    cosA = dot1 / (r1s * dL)
    dot2 = r2_0*BA0 + r2_1*BA1 + r2_2*BA2
    cosB = dot2 / (r2s * dL)

    r_norm = r / rc
    vortexFactor = (vortexStrength/(4.0*pi*r)) * ((r*r) / pow(pow(rc, 2*n) + pow(r, 2*n), 1.0/n))

    if cross_norm < 1e-12:
        return Vout
    Vout[0] = vortexFactor * (cosA - cosB) * cross0 / cross_norm
    Vout[1] = vortexFactor * (cosA - cosB) * cross1 / cross_norm
    Vout[2] = vortexFactor * (cosA - cosB) * cross2 / cross_norm
    return Vout

#----------------------------------------------------------
# calculate_wake_induced_velocity 함수
# WakeMarker: 2D array, 각 행에 여러 3D 포인트(연결된 wake marker)가 저장되어 있음.
# Gamma_TR, Gamma_Shed: 2D array, 각각 trailing 및 shed 와류 강도 (행: wake marker row, 열: panel 개수)
# coloP: (P,3) colocation point들의 위치
# rc: 1D array (panel별 core radius)
#----------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] calculate_wake_induced_velocity(np.ndarray[double, ndim=2] WakeMarker,
                                                                  np.ndarray[double, ndim=2] Gamma_TR,
                                                                  np.ndarray[double, ndim=2] Gamma_Shed,
                                                                  np.ndarray[double, ndim=2] coloP,
                                                                  np.ndarray[double, ndim=1] rc):
    cdef int numWake = WakeMarker.shape[0]
    cdef int numGamma = Gamma_TR.shape[1]
    cdef int numColo = coloP.shape[0]
    if numWake < 2:
        return np.zeros_like(coloP)

    cdef np.ndarray[double, ndim=2] Vout = np.zeros_like(coloP)
    cdef int i, TRWake_Row, TRWake_Col, colIND, colIND2
    cdef double[:] coloP_local
    cdef np.ndarray[double, ndim=1] WakeMarker_A, WakeMarker_B, temp_v, vindTR

    for i in range(numColo):
        coloP_local = coloP[i]
        vindTR = np.zeros(3, dtype=np.float64)
        for TRWake_Row in range(numWake - 1):
            for TRWake_Col in range(numGamma):
                colIND = TRWake_Col * 3
                colIND2 = (TRWake_Col + 1) * 3
                WakeMarker_A = WakeMarker[TRWake_Row, colIND : colIND+3].reshape(3)
                WakeMarker_B = WakeMarker[TRWake_Row + 1, colIND : colIND+3].reshape(3)
                temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, coloP_local,
                                          Gamma_TR[TRWake_Row, TRWake_Col],
                                          rc[TRWake_Col], 1.06)
                vindTR[0] += temp_v[0]
                vindTR[1] += temp_v[1]
                vindTR[2] += temp_v[2]
                if TRWake_Col == numGamma - 1:
                    continue
                WakeMarker_A = WakeMarker[TRWake_Row, colIND : colIND+3].reshape(3).copy()
                WakeMarker_B = WakeMarker[TRWake_Row, colIND2 : colIND2+3].reshape(3).copy()
                temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, coloP_local,
                                          Gamma_Shed[TRWake_Row, TRWake_Col],
                                          rc[TRWake_Col], 1.06)
                vindTR[0] += temp_v[0]
                vindTR[1] += temp_v[1]
                vindTR[2] += temp_v[2]
        Vout[i, 0] = vindTR[0]
        Vout[i, 1] = vindTR[1]
        Vout[i, 2] = vindTR[2]
    return Vout

#----------------------------------------------------------
# calculate_wake_velocity 함수
# WakeMarker: (N, M) 배열 (M는 3의 배수)
# Gamma_TR, Gamma_Shed: trailing/shed 와류 강도 배열
# Gamma_BD: (P, ?) 블레이드 유도 강도 배열, BDpoint, TEPoint: (P,3) 배열
# rc, rcBD: 1D 배열
#----------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2] calculate_wake_velocity(np.ndarray[double, ndim=2] WakeMarker,
                                                          np.ndarray[double, ndim=2] Gamma_TR,
                                                          np.ndarray[double, ndim=2] Gamma_Shed,
                                                          np.ndarray[double, ndim=2] Gamma_BD,
                                                          np.ndarray[double, ndim=2] BDpoint,
                                                          np.ndarray[double, ndim=2] TEPoint,
                                                          np.ndarray[double, ndim=1] rc,
                                                          np.ndarray[double, ndim=1] rcBD):
    cdef int numWake = WakeMarker.shape[0]
    cdef int numGamma = Gamma_TR.shape[1]
    if numWake < 2:
        return np.zeros_like(WakeMarker)

    cdef np.ndarray[double, ndim=2] Vout = np.zeros_like(WakeMarker)
    cdef int marker_wake, marker_geom, TRWake_Row, TRWake_Col, colIND, colIND2, BDWake_Row
    cdef double[:] Marker_local
    cdef np.ndarray[double, ndim=1] WakeMarker_A, WakeMarker_B, temp_v, vindTR

    for marker_wake in range(numWake):
        for marker_geom in range(numGamma):
            colIND = marker_geom * 3
            Marker_local = WakeMarker[marker_wake, colIND : colIND+3]
            vindTR = np.zeros(3, dtype=np.float64)
            for TRWake_Row in range(numWake - 1):
                for TRWake_Col in range(numGamma):
                    colIND = TRWake_Col * 3
                    colIND2 = (TRWake_Col + 1) * 3
                    WakeMarker_A = WakeMarker[TRWake_Row, colIND : colIND+3].reshape(3).copy()
                    WakeMarker_B = WakeMarker[TRWake_Row + 1, colIND : colIND+3].reshape(3).copy()
                    temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                              Gamma_TR[TRWake_Row, TRWake_Col],
                                              rc[TRWake_Col], 1.06)
                    vindTR[0] += temp_v[0]
                    vindTR[1] += temp_v[1]
                    vindTR[2] += temp_v[2]
                    if TRWake_Col == numGamma - 1:
                        continue
                    WakeMarker_A = WakeMarker[TRWake_Row, colIND : colIND+3].reshape(3).copy()
                    WakeMarker_B = WakeMarker[TRWake_Row, colIND2 : colIND2+3].reshape(3).copy()
                    temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                              Gamma_Shed[TRWake_Row, TRWake_Col],
                                              rc[TRWake_Col], 1.06)
                    vindTR[0] += temp_v[0]
                    vindTR[1] += temp_v[1]
                    vindTR[2] += temp_v[2]
            # Blade induced (BD) 부분
            for BDWake_Row in range(Gamma_BD.shape[0]):
                WakeMarker_A = BDpoint[BDWake_Row].reshape(3).copy()
                WakeMarker_B = BDpoint[BDWake_Row+1].reshape(3).copy()
                temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                          Gamma_BD[BDWake_Row, 0],
                                          rcBD[BDWake_Row], 1.06)
                vindTR[0] += temp_v[0]
                vindTR[1] += temp_v[1]
                vindTR[2] += temp_v[2]

                WakeMarker_A = TEPoint[BDWake_Row].reshape(3).copy()
                WakeMarker_B = BDpoint[BDWake_Row].reshape(3).copy()
                temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                          Gamma_BD[BDWake_Row, 0],
                                          rcBD[BDWake_Row], 1.06)
                vindTR[0] += temp_v[0]
                vindTR[1] += temp_v[1]
                vindTR[2] += temp_v[2]

                WakeMarker_A = BDpoint[BDWake_Row+1].reshape(3).copy()
                WakeMarker_B = TEPoint[BDWake_Row+1].reshape(3).copy()
                temp_v = Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                          Gamma_BD[BDWake_Row, 0],
                                          rcBD[BDWake_Row], 1.06)
                vindTR[0] += temp_v[0]
                vindTR[1] += temp_v[1]
                vindTR[2] += temp_v[2]
            Vout[marker_wake, marker_geom*3 + 0] = vindTR[0]
            Vout[marker_wake, marker_geom*3 + 1] = vindTR[1]
            Vout[marker_wake, marker_geom*3 + 2] = vindTR[2]
    return Vout
