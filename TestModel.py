import joblib
import time
import numpy as np
from numba import jit
print("read ...")

Section1_CL = joblib.load('Section1_CL.pkl')
Section1_CD = joblib.load('Section1_CD.pkl')
Section1_CL_near = joblib.load('Section1_CL_near.pkl')
Section1_CD_near = joblib.load('Section1_CD_near.pkl')
Section1_CL(0,0,0)
Section1_CD(0,0,0)
Section1_CL_near(0,0,0)
Section1_CD_near(0,0,0)
Section2_CL = joblib.load('Section2_CL.pkl')
Section2_CD = joblib.load('Section2_CD.pkl')
Section2_CL_near = joblib.load('Section2_CL_near.pkl')
Section2_CD_near = joblib.load('Section2_CD_near.pkl')
Section2_CL(0,0,0)
Section2_CD(0,0,0)
Section2_CL_near(0,0,0)
Section2_CD_near(0,0,0)

BLD_CL = joblib.load('BLD_CL.pkl')
BLD_CD = joblib.load('BLD_CD.pkl')
BLD_CL_near = joblib.load('BLD_CL_near.pkl')
BLD_CD_near = joblib.load('BLD_CD_near.pkl')
BLD_CL(0,0,0)
BLD_CD(0,0,0)
BLD_CL_near(0,0,0)
BLD_CD_near(0,0,0)


print("read done")
#@jit
def Aeromodel(model,arg1,arg2,arg3):
    cl=0.0
    cd=0.0
    if model == 1:
        cl = Section1_CL(arg1, arg2, arg3)
        cd = Section1_CD(arg1, arg2, arg3)
        if cl > 100:
            cl = Section1_CL_near(arg1, arg2, arg3)
            cd = Section1_CD_near(arg1, arg2, arg3)
        return np.array([cl, cd])

    if model == 2:
        cl = Section2_CL(arg1, arg2, arg3)
        cd = Section2_CD(arg1, arg2, arg3)
        if cl > 100:
            cl = Section2_CL_near(arg1, arg2, arg3)
            cd = Section2_CD_near(arg1, arg2, arg3)
        return np.array([cl, cd])

    else:
        cl = BLD_CL(arg1, arg2, arg3)
        cd = BLD_CD(arg1, arg2, arg3)
        if cl > 100:
            cl = BLD_CL_near(arg1, arg2, arg3)
            cd = BLD_CD_near(arg1, arg2, arg3)
        return np.array([cl, cd])

def Aeromodel2(model,model_near,arg1,arg2,arg3):
    output = model(arg1, arg2, arg3)

    if output > 100:
        output = model_near(arg1, arg2, arg3)

    return output


print()
print()
start= time.time()
cl=Aeromodel2(Section1_CL,Section1_CL_near,10,50000,10)
cd=Aeromodel2(Section1_CD,Section1_CD_near,10,50000,10)
print ("CL : ",cl,"CD :",cd)
print(f"{time.time() - start:.4f} sec")
start= time.time()
cl=Aeromodel2(Section2_CL,Section2_CL_near,10,50000,10)
cd=Aeromodel2(Section2_CD,Section2_CD_near,10,50000,10)
print ("CL : ",cl,"CD :",cd)

print(f"{time.time() - start:.4f} sec")
start= time.time()
cl=Aeromodel2(BLD_CL,BLD_CL_near,10,50000,10)
cd=Aeromodel2(BLD_CD,BLD_CD_near,10,50000,10)
print ("CL : ",cl,"CD :",cd)

print(f"{time.time() - start:.4f} sec")

print()
print()
start= time.time()
cl=Aeromodel2(Section1_CL,Section1_CL_near,12,50000,10)
cd=Aeromodel2(Section1_CD,Section1_CD_near,12,50000,10)
print ("CL : ",cl,"CD :",cd)
print(f"{time.time() - start:.4f} sec")
start= time.time()
cl=Aeromodel2(Section2_CL,Section2_CL_near,12,50000,10)
cd=Aeromodel2(Section2_CD,Section2_CD_near,12,50000,10)
print ("CL : ",cl,"CD :",cd)

print(f"{time.time() - start:.4f} sec")
start= time.time()
cl=Aeromodel2(BLD_CL,BLD_CL_near,12,50000,10)
cd=Aeromodel2(BLD_CD,BLD_CD_near,12,50000,10)
print ("CL : ",cl,"CD :",cd)

print(f"{time.time() - start:.4f} sec")

print()
print()
start= time.time()
data=Aeromodel(1,7,100000,5)
print(f"{time.time() - start:.4f} sec")
start= time.time()
data2=Aeromodel(2,7,100000,5)
print(f"{time.time() - start:.4f} sec")
start= time.time()
data3=Aeromodel(3,7,100000,5)
print(f"{time.time() - start:.4f} sec")

print("Interpolated Cl:", data[0])
print("Interpolated Cd:", data[1])
print("Interpolated Cl:", data2[0])
print("Interpolated Cd:", data2[1])
print("Interpolated Cl:", data3[0])
print("Interpolated Cd:", data3[1])

# (3) 특정 샘플 값에서 Cl과 Cd 예측
sampleThk = 10
sampleRE = 50000
sampleAOA = 10

start= time.time()
cl = Section1_CL(sampleThk, sampleRE, sampleAOA)
cd = Section1_CD(sampleThk, sampleRE, sampleAOA)

print("Interpolated Cl:", cl)
print("Interpolated Cd:", cd)

cl = Section2_CL_near(sampleThk, sampleRE, 10)
cd = Section2_CD_near(sampleThk, sampleRE, 100)

print("Interpolated Cl:", cl)
print("Interpolated Cd:", cd)
print(f"{time.time() - start:.4f} sec")

def Aeromodel(model,arg1,arg2,arg3):
    cl=0.0
    cd=0.0
    cl0 = BLD_CL(arg1, arg2, arg3)
    cd0 = BLD_CD(arg1, arg2, arg3)
    if cl0 > 100:
        cl0 = BLD_CL_near(arg1, arg2, arg3)
        cd0 = BLD_CD_near(arg1, arg2, arg3)
    cl1 = Section1_CL(arg1, arg2, arg3)
    cd1 = Section1_CD(arg1, arg2, arg3)
    if cl1 > 100:
        cl1 = Section1_CL_near(arg1, arg2, arg3)
        cd1 = Section1_CD_near(arg1, arg2, arg3)
    cl2 = Section2_CL(arg1, arg2, arg3)
    cd2 = Section2_CD(arg1, arg2, arg3)
    if cl2 > 100:
        cl2 = Section2_CL_near(arg1, arg2, arg3)
        cd2 = Section2_CD_near(arg1, arg2, arg3)



    return np.array([[cl0, cd0], [cl1, cd1], [cl2, cd2]])



print()
print()
start= time.time()
data=Aeromodel(1,7,100000,5)
print(f"{time.time() - start:.4f} sec")



print("Interpolated Cl:", data[0,1])
print("Interpolated Cd:", data[0,1])
print("Interpolated Cl:", data[1,0])
print("Interpolated Cd:", data[1,1])
print("Interpolated Cl:", data[2,0])
print("Interpolated Cd:", data[2,1])