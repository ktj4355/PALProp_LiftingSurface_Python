import os
import time
from numba import jit
import numpy as np
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
import joblib

# 데이터 저장을 위한 리스트
DB_Aero = []  # [thick, RE, AoA, Cl, Cd]
DBind = []  # [RE, Mach]
Total_DB = []

# "Section1" 폴더 내 파일 리스트 가져오기
folder_path = os.path.join(os.getcwd(), "aerodb", "section1")

file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 파일 읽기 및 데이터 처리
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    #print(file_path)
    # 파일명에서 정보 추출 (예: E63_4_27_T1_Re0.050_M0.20_N7.0.txt)
    name_split = file_name.split('_')
    airfoil_name = name_split[0]
    thick_major = int(name_split[1])
    thick_minor = int(name_split[2])
    thick = thick_major + thick_minor * 0.01
    #print(name_split)

    # 파일 내용 읽기
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 8번째 줄에서 Mach, Re 값 추출
    setting = list( lines[7].split())



    local_mach = float(setting[2])
    local_re = float(setting[5]) * (10 ** float(setting[7]))
    #print(local_mach)
    #print(local_re)

    # 12번째 줄부터 데이터 읽기
    data = []
    for line in lines[11:-3]:
        #print(line.split())

        values = list(map(float, line.split()))
        data.append(values)

    if data:
        aero_db_local = np.array(data)

        aero_db_local = aero_db_local[:, 0:4]  # [aoa Cl, Cd, Cd_p]

        # 보간 테이블에 추가
        mach_col =  np.full((aero_db_local.shape[0], 1), local_mach)
        thick_col = np.full((aero_db_local.shape[0], 1), thick)
        re_col = np.full((aero_db_local.shape[0], 1), local_re)

        db_aero_local = np.hstack((mach_col,thick_col,re_col,aero_db_local[:, :3]))  # [thick, Re,alpha Cl, Cd]
       # print (db_aero_local)

        # 데이터 저장
        DB_Aero.append(db_aero_local)
        DBind.append([local_re, local_mach])
        Total_DB.append(db_aero_local)

# numpy 배열로 변환
DB_Aero = np.vstack(DB_Aero) if DB_Aero else np.array([])
DBind = np.array(DBind)
Total_DB = np.vstack(Total_DB) if Total_DB else np.array([])


points = Total_DB[:, 1:4]  # [thick, RE, AoA] → (x, y, z)
# mach number 사용시
#points = Total_DB[:, 0:4]

cl_values = Total_DB[:, 4]  # Cl 값
cd_values = Total_DB[:, 5]  # Cd 값
#print(points)
# (2) 보간 함수 생성 (Matlab의 scatteredInterpolant와 동일한 방식)
Section1_CL = LinearNDInterpolator(points, cl_values,fill_value=999)  # Cl 보간 함수
Section1_CD = LinearNDInterpolator(points, cd_values,fill_value=999)  # Cd 보간 함수
Section1_CL_near = NearestNDInterpolator(points, cl_values)  # Cl 보간 함수
Section1_CD_near = NearestNDInterpolator(points, cd_values)  # Cd 보간 함수
joblib.dump(Section1_CL_near, 'Section1_CL_near.pkl')
joblib.dump(Section1_CD_near, 'Section1_CD_near.pkl')
joblib.dump(Section1_CL, 'Section1_CL.pkl')
joblib.dump(Section1_CD, 'Section1_CD.pkl')




# 초기화
DB_Aero = []  # [thick, RE, AoA, Cl, Cd]
DBind = []  # [RE, Mach]
Total_DB = []
# "Section1" 폴더 내 파일 리스트 가져오기
folder_path = os.path.join(os.getcwd(), "aerodb", "section2")
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 파일 읽기 및 데이터 처리
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    #print(file_path)
    # 파일명에서 정보 추출 (예: E63_4_27_T1_Re0.050_M0.20_N7.0.txt)
    name_split = file_name.split('_')
    airfoil_name = name_split[0]
    thick_major = int(name_split[1])
    thick_minor = int(name_split[2])
    thick = thick_major + thick_minor * 0.01
    #print(name_split)

    # 파일 내용 읽기
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 8번째 줄에서 Mach, Re 값 추출
    setting = list( lines[7].split())



    local_mach = float(setting[2])
    local_re = float(setting[5]) * (10 ** float(setting[7]))
    #print(local_mach)
    #print(local_re)

    # 12번째 줄부터 데이터 읽기
    data = []
    for line in lines[11:-3]:
        #print(line.split())

        values = list(map(float, line.split()))
        data.append(values)

    if data:
        aero_db_local = np.array(data)

        aero_db_local = aero_db_local[:, 0:4]  # [aoa Cl, Cd, Cd_p]

        # 보간 테이블에 추가
        mach_col =  np.full((aero_db_local.shape[0], 1), local_mach)
        thick_col = np.full((aero_db_local.shape[0], 1), thick)
        re_col = np.full((aero_db_local.shape[0], 1), local_re)

        db_aero_local = np.hstack((mach_col,thick_col,re_col,aero_db_local[:, :3]))  # [thick, Re,alpha Cl, Cd]
        #print (db_aero_local)

        # 데이터 저장
        DB_Aero.append(db_aero_local)
        DBind.append([local_re, local_mach])
        Total_DB.append(db_aero_local)

# numpy 배열로 변환
DB_Aero = np.vstack(DB_Aero) if DB_Aero else np.array([])
DBind = np.array(DBind)
Total_DB = np.vstack(Total_DB) if Total_DB else np.array([])


points = Total_DB[:, 1:4]  # [thick, RE, AoA] → (x, y, z)
# mach number 사용시
#points = Total_DB[:, 0:4]

cl_values = Total_DB[:, 4]  # Cl 값
cd_values = Total_DB[:, 5]  # Cd 값
#print(points)
# (2) 보간 함수 생성 (Matlab의 scatteredInterpolant와 동일한 방식)
Section2_CL_near = NearestNDInterpolator(points, cl_values)  # Cl 보간 함수
Section2_CD_near = NearestNDInterpolator(points, cd_values)  # Cd 보간 함수
Section2_CL = LinearNDInterpolator(points, cl_values,fill_value=999)  # Cl 보간 함수
Section2_CD = LinearNDInterpolator(points, cd_values,fill_value=999)  # Cd 보간 함수
joblib.dump(Section2_CL, 'Section2_CL.pkl')
joblib.dump(Section2_CD, 'Section2_CD.pkl')
joblib.dump(Section2_CL_near, 'Section2_CL_near.pkl')
joblib.dump(Section2_CD_near, 'Section2_CD_near.pkl')


# 초기화
DB_Aero = []  # [thick, RE, AoA, Cl, Cd]
DBind = []  # [RE, Mach]
Total_DB = []

# "Section1" 폴더 내 파일 리스트 가져오기
folder_path = os.path.join(os.getcwd(), "aerodb", "section2")

file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 파일 읽기 및 데이터 처리
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    #print(file_path)
    # 파일명에서 정보 추출 (예: E63_4_27_T1_Re0.050_M0.20_N7.0.txt)
    name_split = file_name.split('_')
    airfoil_name = name_split[0]
    thick_major = int(name_split[1])
    thick_minor = int(name_split[2])
    thick = thick_major + thick_minor * 0.01
    #print(name_split)

    # 파일 내용 읽기
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 8번째 줄에서 Mach, Re 값 추출
    setting = list( lines[7].split())



    local_mach = float(setting[2])
    local_re = float(setting[5]) * (10 ** float(setting[7]))
    #print(local_mach)
    #print(local_re)

    # 12번째 줄부터 데이터 읽기
    data = []
    for line in lines[11:-3]:
        #print(line.split())

        values = list(map(float, line.split()))
        data.append(values)

    if data:
        aero_db_local = np.array(data)

        aero_db_local = aero_db_local[:, 0:4]  # [aoa Cl, Cd, Cd_p]

        # 보간 테이블에 추가
        mach_col =  np.full((aero_db_local.shape[0], 1), local_mach)
        thick_col = np.full((aero_db_local.shape[0], 1), thick)
        re_col = np.full((aero_db_local.shape[0], 1), local_re)

        db_aero_local = np.hstack((mach_col,thick_col,re_col,aero_db_local[:, :3]))  # [thick, Re,alpha Cl, Cd]
        #print (db_aero_local)

        # 데이터 저장
        DB_Aero.append(db_aero_local)
        DBind.append([local_re, local_mach])
        Total_DB.append(db_aero_local)

# numpy 배열로 변환
DB_Aero = np.vstack(DB_Aero) if DB_Aero else np.array([])
DBind = np.array(DBind)
Total_DB = np.vstack(Total_DB) if Total_DB else np.array([])


points = Total_DB[:, 1:4]  # [thick, RE, AoA] → (x, y, z)
# mach number 사용시
#points = Total_DB[:, 0:4]

cl_values = Total_DB[:, 4]  # Cl 값
cd_values = Total_DB[:, 5]  # Cd 값

# (2) 보간 함수 생성 (Matlab의 scatteredInterpolant와 동일한 방식)
BLD_CL = LinearNDInterpolator(points, cl_values,fill_value=999)  # Cl 보간 함수
BLD_CD = LinearNDInterpolator(points, cd_values,fill_value=999)  # Cd 보간 함수
joblib.dump(BLD_CL, 'BLD_CL.pkl')
joblib.dump(BLD_CD, 'BLD_CD.pkl')
BLD_CL_near = NearestNDInterpolator(points, cl_values,)  # Cl 보간 함수
BLD_CD_near = NearestNDInterpolator(points, cd_values)  # Cd 보간 함수
joblib.dump(BLD_CL_near, 'BLD_CL_near.pkl')
joblib.dump(BLD_CD_near, 'BLD_CD_near.pkl')

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