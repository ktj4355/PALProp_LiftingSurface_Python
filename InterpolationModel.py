import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
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
        print (db_aero_local)

        # 데이터 저장
        DB_Aero.append(db_aero_local)
        DBind.append([local_re, local_mach])
        Total_DB.append(db_aero_local)

# numpy 배열로 변환
DB_Aero = np.vstack(DB_Aero) if DB_Aero else np.array([])
DBind = np.array(DBind)
Total_DB = np.vstack(Total_DB) if Total_DB else np.array([])

# 결과 출력
#print(Total_DB)

points = Total_DB[:, 1:4]  # [thick, RE, AoA] → (x, y, z)
# mach number 사용시
#points = Total_DB[:, 0:4]

cl_values = Total_DB[:, 4]  # Cl 값
cd_values = Total_DB[:, 5]  # Cd 값
print(points)
# (2) 보간 함수 생성 (Matlab의 scatteredInterpolant와 동일한 방식)
Section1_CL = LinearNDInterpolator(points, cl_values)  # Cl 보간 함수
Section1_CD = LinearNDInterpolator(points, cd_values)  # Cd 보간 함수

# (3) 특정 샘플 값에서 Cl과 Cd 예측
sampleThk = 10
sampleRE = 50000
sampleAOA = 10

cl = Section1_CL(sampleThk, sampleRE, sampleAOA)
cd = Section1_CD(sampleThk, sampleRE, sampleAOA)

print("Interpolated Cl:", cl)
print("Interpolated Cd:", cd)

cl = Section1_CL(sampleThk, sampleRE, -10)
cd = Section1_CD(sampleThk, sampleRE, -10)

print("Interpolated Cl:", cl)
print("Interpolated Cd:", cd)

joblib.dump(Section1_CL, 'Section1_CL.pkl')
joblib.dump(Section1_CD, 'Section1_CD.pkl')