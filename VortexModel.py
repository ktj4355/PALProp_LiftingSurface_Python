import sys
import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as LA
from functools import cache
from numba import njit,jit
#@cache
@njit(cache=True)
def Vortex_Scully(A, B, ColocationPoint, vortexStrength, rc):
    r1 = (ColocationPoint - A).copy()
    r2 = (ColocationPoint - B).copy()
    dL = abs(LA.norm(A - B))
    r = LA.norm(np.cross(r1, r2)) / dL
    Vout = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # dtype=np.float64로 수정

    if r < 0.000001 or vortexStrength < 0.00001:
        return Vout
    r1s = LA.norm(r1)
    r2s = LA.norm(r2)

    cosA = np.dot(r1, (B - A)) / (r1s * dL)
    cosB = np.dot(r2, (B - A)) / (r2s * dL)


    r_norm = r / rc
    a = 1.25643

    vortexFactor = (vortexStrength / (4 * np.pi * r)) * ((r_norm ** 2) / (1 + (r_norm ** 2)))
    Vout = vortexFactor * (cosA - cosB) * np.cross(r1, r2) / (LA.norm(np.cross(r1, r2)))
    return Vout
def Vortex_rankin(A,B,ColocationPoint,vortexStrength,rc):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    r1 = (ColocationPoint - A).copy()
    r2 = (ColocationPoint - B).copy()

    r1s = LA.norm(r1)
    r2s = LA.norm(r2)
    dL = abs(LA.norm(A - B))

    cosA = np.dot(r1, (B - A)) / (r1s * dL)
    cosB = np.dot(r2, (B - A)) / (r2s * dL)
    r = LA.norm(np.cross(r1, r2)) / dL
    if r < 0.000001:
        Vout = np.array([0, 0, 0])
        return Vout.copy()


    r_norm = r / rc
    if r_norm < 1:
        vortexFactor = (vortexStrength/ (4 * np.pi * r)) * (r_norm ** 2)
        Vout = vortexFactor * (cosA - cosB) * np.cross(r1, r2)/ (LA.norm(np.cross(r1, r2)))
    else:
        vortexFactor = vortexStrength/ (4 * np.pi * r)
        Vout = vortexFactor * (cosA - cosB) * np.cross(r1, r2)/ (LA.norm(np.cross(r1, r2)))
    return Vout.copy()
def Vortex_lambOseen(A,B,ColocationPoint,vortexStrength,rc):
    r1 = (ColocationPoint - A).copy()
    r2 = (ColocationPoint - B).copy()

    r1s = LA.norm(r1)
    r2s = LA.norm(r2)
    dL = abs(LA.norm(A - B))

    cosA = np.dot(r1, (B - A)) / (r1s * dL)
    cosB = np.dot(r2, (B - A)) / (r2s * dL)
    r = LA.norm(np.cross(r1, r2)) / dL
    if r < 0.000001:
        Vout = np.array([0, 0, 0])
        return Vout.copy()

    r_norm = r / rc
    a = 1.25643
    vortexFactor = (vortexStrength/ (4 * np.pi * r))  *(1-np.exp(-a*r_norm*r_norm))
    Vout = vortexFactor * (cosA - cosB) * np.cross(r1, r2)/ (LA.norm(np.cross(r1, r2)))
    return Vout.copy()
def Vortex_bio(A,B,ColocationPoint,vortexStrength,rc):
    r1 = (ColocationPoint - A).copy()
    r2 = (ColocationPoint - B).copy()

    r1s = LA.norm(r1)
    r2s = LA.norm(r2)
    dL = abs(LA.norm(A - B))

    cosA = np.dot(r1, (B - A)) / (r1s * dL)
    cosB = np.dot(r2, (B - A)) / (r2s * dL)
    r = LA.norm(np.cross(r1, r2)) / dL
    if r < 0.000001:
        Vout = np.array([0, 0, 0])
        return Vout.copy()
    vortexFactor = vortexStrength / (4 * np.pi * r)
    Vout = vortexFactor * (cosA - cosB) * np.cross(r1, r2) / (LA.norm(np.cross(r1, r2)))
    return Vout.copy()
def Vortex_Vatistas(A,B,ColocationPoint,vortexStrength,rc,n):
    r1 = (ColocationPoint - A).copy()
    r2 = (ColocationPoint - B).copy()

    r1s = LA.norm(r1)
    r2s = LA.norm(r2)
    dL = abs(LA.norm(A - B))

    cosA = np.dot(r1, (B - A)) / (r1s * dL)
    cosB = np.dot(r2, (B - A)) / (r2s * dL)
    r = LA.norm(np.cross(r1, r2)) / dL
    if r < 0.000001:
        Vout = np.array([0, 0, 0])
        return Vout.copy()

    r_norm = r / rc
    a = 1.25643

    vortexFactor=(vortexStrength/(4*np.pi*r))*((r**2)/( (rc**(2*n)+r**(2*n))**(1/n) ))
    Vout = vortexFactor * (cosA - cosB) * np.cross(r1, r2) / (LA.norm(np.cross(r1, r2)))
    return Vout.copy()

def PlotVortex():
    x=np.linspace(0.02,0.6,100)
    A=np.array([-1, 0, 0])
    B=np.array([1, 0, 0])
    Vel=np.zeros([x.shape[0],7])
    for xind in range(x.shape[0]):
        ColocationPoint=[0,x[xind],0]
        vortexStrength=1
        rc=0.2
        Vel[xind,:]=np.array([LA.norm(Vortex_bio(A,B,ColocationPoint,vortexStrength,rc)),
                               LA.norm(Vortex_rankin(A,B,ColocationPoint,vortexStrength,rc)),
                               LA.norm(Vortex_lambOseen(A,B,ColocationPoint,vortexStrength,rc)),
                               LA.norm(Vortex_Scully(A,B,ColocationPoint,vortexStrength,rc)),
                               LA.norm(Vortex_Vatistas(A,B,ColocationPoint,vortexStrength,rc,1.06)),
                               LA.norm(Vortex_Vatistas(A,B,ColocationPoint,vortexStrength,rc,2)),
                               LA.norm(Vortex_Vatistas(A,B,ColocationPoint,vortexStrength,rc,3))]).copy()

    print(Vel)


    #plt.axes().set_aspect('equal')
    plt.plot(x,Vel)
    plt.ylim(0,2)
    plt.show()