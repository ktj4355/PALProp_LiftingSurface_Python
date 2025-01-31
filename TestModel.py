import joblib
import time

Section1_CL_loaded = joblib.load('Section1_CL.pkl')
Section1_CD_loaded = joblib.load('Section1_CD.pkl')
start= time.time()
cl = Section1_CL_loaded(10, 50000, 0)
cd=Section1_CD_loaded(10, 50000, 0)
print(f"{time.time() - start:.4f} sec")
print("Interpolated Cl:", cl)
print("Interpolated Cl:", cd)