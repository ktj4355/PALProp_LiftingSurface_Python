import random
import time

import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np

# ===============================
# Aerodynamic Model Initialization
# ===============================
# 각 구간(Section1, Section2, Blade)의 양력/항력 계수 모델을 joblib를 이용하여 불러옴
print("read ...")
Section1_CL = joblib.load('Section1_CL.pkl')
Section1_CD = joblib.load('Section1_CD.pkl')
Section1_CL_near = joblib.load('Section1_CL_near.pkl')
Section1_CD_near = joblib.load('Section1_CD_near.pkl')
# 모델의 초기 호출(예: 캐싱이나 확인 목적)
Section1_CL(0, 0, 0)
Section1_CD(0, 0, 0)
Section1_CL_near(0, 0, 0)
Section1_CD_near(0, 0, 0)

Section2_CL = joblib.load('Section2_CL.pkl')
Section2_CD = joblib.load('Section2_CD.pkl')
Section2_CL_near = joblib.load('Section2_CL_near.pkl')
Section2_CD_near = joblib.load('Section2_CD_near.pkl')
Section2_CL(0, 0, 0)
Section2_CD(0, 0, 0)
Section2_CL_near(0, 0, 0)
Section2_CD_near(0, 0, 0)

BLD_CL = joblib.load('BLD_CL.pkl')
BLD_CD = joblib.load('BLD_CD.pkl')
BLD_CL_near = joblib.load('BLD_CL_near.pkl')
BLD_CD_near = joblib.load('BLD_CD_near.pkl')
BLD_CL(0, 0, 0)
BLD_CD(0, 0, 0)
BLD_CL_near(0, 0, 0)
BLD_CD_near(0, 0, 0)


# ===============================
# Aeromodel 함수: 주어진 입력값에 따라 적절한 모델을 선택하여
# 양력(cl)과 항력(cd)를 계산함
# ===============================
def Aeromodel(model, arg1, arg2, arg3):
    """
    Parameters:
      - model: 1, 2, 또는 그 외의 값에 따라 서로 다른 모델을 선택
      - arg1: 두께(thickness) 등 물리적 파라미터
      - arg2: Reynolds number (RE)
      - arg3: Angle of Attack (AOA)

    Returns:
      - np.array([cl, cd]): 계산된 양력 및 항력 계수
    """
    cl = 0.0
    cd = 0.0
    if model == 1:
        cl = Section1_CL(arg1, arg2, arg3)
        cd = Section1_CD(arg1, arg2, arg3)
        # cl 값이 매우 클 경우 near 모델 사용
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


# ===============================
# 유틸리티: 각도를 인자로 하는 삼각함수 (입력 단위: 도)
# ===============================
def sind(angle):
    """angle(도)의 사인값 반환"""
    return np.sin(np.deg2rad(angle))
def cosd(angle):
    """angle(도)의 코사인값 반환"""
    return np.cos(np.deg2rad(angle))
def tand(angle):
    """angle(도)의 탄젠트값 반환"""
    return np.tan(np.deg2rad(angle))


# ===============================
# 시뮬레이션 파라미터 설정
# ===============================
start = time.time()

# Prop 설치 조건
TiltAngle_Fixedwing = 0  # 고정익의 경우, 0도: 전방향
# Drone의 경우, 90°에서 빼주는 형태로 계산 (그러나 TiltAngle_drone는 이후 사용되지 않음)
TiltAngle_drone = 90 - TiltAngle_Fixedwing  # -> 사용하지 않음
Prop_TiltAngle = 30  # Drone 기준 프로펠러 틸트각 (피치)
Prop_TiltPhi = 45  # 프로펠러의 방위각(azimuth)
R = 1.25  # 시각화 기준 반경
# Vehicle 조건: 지면 기준 속도 (단위는 임의)
vehicle_speed_x = 0  # 전방속도 (phi = 0)
vehicle_speed_y = 0  # 측면속도 (phi = 90)
vehicle_speed_z = -200  # 상승/하강속도 (음수이면 하강, 주석은 Ascend라고 되어 있으나 실제 값은 음수)
G_Vehicle_Speed_vec = np.array([vehicle_speed_x, vehicle_speed_y, vehicle_speed_z])
# Inflow 벡터는 차량 속도의 반대 방향
G_Inflow_Vehicle_vec = -G_Vehicle_Speed_vec
rpm = 1150  # 프로펠러 회전수 (rpm)
n = rpm / 60  # 회전수를 초당 회전수로 변환
rR = np.linspace(0, 1, 10)  # 블레이드의 정규화된 반경 좌표

# 유도(induced) 속도 (예시 값)
G_v_ind_col = np.array([0, 0, -10])
# ===============================
# 프로펠러 회전행렬 설정
# ===============================
# z축(방위각)와 y축(틸트각)을 이용하여 회전행렬을 구성
zAxisAngle = Prop_TiltPhi  # 방위각 (z축 회전)
yAxisAngle = Prop_TiltAngle  # 피치 각도 (y축 회전)

# y축 회전 (피치 회전)
pitch = np.deg2rad(yAxisAngle)
Rp = np.array([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
])

# z축 회전 (요 회전)
yaw = np.deg2rad(zAxisAngle)
Ry = np.array([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
])
# 두 회전을 결합하여 프로펠러의 전체 회전행렬 생성
Rot_mat = Ry @ Rp

# ===============================
# 지면 기준 단위벡터
# ===============================
groundUnit_X = np.array([1, 0, 0])
groundUnit_Y = np.array([0, 1, 0])
groundUnit_Z = np.array([0, 0, 1])

# ===============================
# Disk (프로펠러) 프레임 설정
# ===============================
# 프로펠러의 회전행렬을 이용하여 disk 프레임의 단위벡터를 정의
Unit_Disk_tanX = Rot_mat @ groundUnit_X  # 디스크 접선 방향 (azimuth 0)
Unit_Disk_tanY = Rot_mat @ groundUnit_Y  # 디스크 접선 방향 (azimuth 90)
Unit_Disk_Axis = Rot_mat @ groundUnit_Z  # 디스크의 법선 (축)

# ===============================
# Disk 프레임에서의 Inflow 성분 계산
# ===============================
G_Inflow_Disk_Axis = np.dot(G_Inflow_Vehicle_vec, Unit_Disk_Axis)
G_Inflow_Disk_tanX = np.dot(G_Inflow_Vehicle_vec, Unit_Disk_tanX)
G_Inflow_Disk_tanY = np.dot(G_Inflow_Vehicle_vec, Unit_Disk_tanY)

# 디스크 프레임 좌표계에서의 Inflow 벡터 (접선 성분 2개, 축 성분 1개)
D_Inflow_Disk_vec = np.array([G_Inflow_Disk_tanX, G_Inflow_Disk_tanY, G_Inflow_Disk_Axis])

# 아래 3개의 변수는 계산되지만 이후 사용되지 않으므로 주석 처리함
# G_Inflow_Disk_Axis_vec = G_Inflow_Disk_Axis * Unit_Disk_Axis
# G_Inflow_Disk_tanX_vec = G_Inflow_Disk_tanX * Unit_Disk_tanX
# G_Inflow_Disk_tanY_vec = G_Inflow_Disk_tanY * Unit_Disk_tanY

# ===============================
# 3D 시각화를 위한 플롯 설정
# ===============================
fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')

ax.axis([-R, R, -R, R, -R, R])

# 프로펠러 방향을 나타내는 기준 선 생성
azm = np.linspace(0, 360, 36)
refposition = Rot_mat @ np.array([R, 0, 0])
refposition = np.vstack((np.array([0, 0, 0]), refposition))
print("refposition : ", refposition)
ax.plot(refposition[:, 0], refposition[:, 1], refposition[:, 2])

# ===============================
# 각 방위각(azimuth)에 대해 블레이드 로드 계산 및 시각화
# ===============================
for azmIdx in range(len(azm)):
    Now_Azimuth_Angle = azm[azmIdx]
    # 현재 방위각에 따른 회전행렬 생성 (도 단위 사용)
    RAzimuth = np.array([
        [cosd(Now_Azimuth_Angle), -sind(Now_Azimuth_Angle), 0],
        [sind(Now_Azimuth_Angle), cosd(Now_Azimuth_Angle), 0],
        [0, 0, 1]
    ])

    # 블레이드의 로컬 좌표계 단위벡터 설정
    # BLP_unit_chord_neg: 블레이드의 코드(날개 앞면) 반대 방향 (항력 관련)
    # BLP_unit_span: 블레이드의 날개 방향 (spanwise)
    # BLP_unit_axis: 블레이드의 축 (추력 방향)
    BLP_unit_chord_neg = RAzimuth @ np.array([0, 1, 0])
    BLP_unit_span = RAzimuth @ np.array([1, 0, 0])
    BLP_unit_axis = RAzimuth @ np.array([0, 0, 1])

    # 아래 중복 정의된 변수들은 사용되지 않으므로 주석 처리함
    # BLP_unit_x = RAzimuth @ np.array([1, 0, 0])  # BLP_unit_span과 동일
    # BLP_unit_y = RAzimuth @ np.array([0, 1, 0])  # BLP_unit_chord_neg과 동일
    # BLP_unit_z = RAzimuth @ np.array([0, 0, 1])  # BLP_unit_axis와 동일

    # 프로펠러 회전행렬을 적용하여, 블레이드 좌표계를 지면 좌표계로 변환
    BLP_G_unit_chord_neg = Rot_mat @ BLP_unit_chord_neg
    BLP_G_unit_span = Rot_mat @ BLP_unit_span
    BLP_G_unit_axis = Rot_mat @ BLP_unit_axis

    # 블레이드의 로컬 좌표계에서 Inflow 성분 계산
    BLP_inflow_Chord_neg = np.dot(D_Inflow_Disk_vec, BLP_unit_chord_neg)
    BLP_inflow_Span = np.dot(D_Inflow_Disk_vec, BLP_unit_span)
    BLP_inflow_Axis = np.dot(D_Inflow_Disk_vec, BLP_unit_axis)

    # 아래의 블레이드 Inflow 벡터들은 이후 사용되지 않으므로 주석 처리함
    # BLP_inflow_chord_neg_vec = BLP_unit_chord_neg * BLP_inflow_Chord_neg
    # BLP_inflow_span_vec = BLP_unit_span * BLP_inflow_Span
    # BLP_inflow_axis_vec = BLP_unit_axis * BLP_inflow_Axis

    # ===============================
    # Blade Load Model 파라미터 및 Inflow 설정
    # ===============================
    # beta, Local_chold 등 일부 파라미터는 계산되나 사용되지 않으므로 주석 처리함
    # beta = 30
    # Local_chold = 0.1


    # 디스크 프레임에서의 유도 속도 성분 계산
    D_Vax_induce = np.dot(G_v_ind_col, Unit_Disk_Axis)
    D_Vtx_induce = np.dot(G_v_ind_col, Unit_Disk_tanX)
    D_Vty_induce = np.dot(G_v_ind_col, Unit_Disk_tanY)
    Disk_Vinduced_Vector = np.array([D_Vtx_induce, D_Vty_induce, D_Vax_induce])

    # 블레이드 로컬 좌표계에서 유도 속도 성분 계산
    BLP_Vinduced_chord_neg = np.dot(Disk_Vinduced_Vector, BLP_unit_chord_neg)
    BLP_Vinduced_span = np.dot(Disk_Vinduced_Vector, BLP_unit_span)
    BLP_Vinduced_axis = np.dot(Disk_Vinduced_Vector, BLP_unit_axis)

    # ===============================
    # 블레이드 상의 각 반경 위치에서의 계산
    # ===============================
    for idx in range(len(rR)):
        r = R * rR[idx]  # 실제 블레이드의 반경 위치

        # 해당 반경에서의 회전에 의한 접선 속도 계산
        Vt_rot = (2 * np.pi * n * r) * BLP_G_unit_chord_neg

        # 회전, 자유 유동, 유도 속도의 각 성분 계산
        L_Vt_inflow_Axis = 0 * BLP_G_unit_axis  # 축 방향은 0
        L_Vt_inflow_Chord = -Vt_rot  # 코드 방향(항력 관련)
        L_Vt_inflow_Span = 0 * BLP_G_unit_span  # span 방향은 0

        # 자유 유동(차량 속도)의 각 성분 (지면 기준)
        L_vFree_Axis = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_axis) * BLP_G_unit_axis
        L_vFree_Chord = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_chord_neg) * BLP_G_unit_chord_neg
        L_vFree_Span = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_span) * BLP_G_unit_span

        # 유도 속도의 각 성분 (지면 기준)
        L_Vinduced_Axis = np.dot(G_v_ind_col, BLP_G_unit_axis) * BLP_G_unit_axis
        L_Vinduced_Chord = np.dot(G_v_ind_col, BLP_G_unit_chord_neg) * BLP_G_unit_chord_neg
        L_Vinduced_Span = np.dot(G_v_ind_col, BLP_G_unit_span) * BLP_G_unit_span

        # 각 성분의 합산: 블레이드 요소에서의 전체 유동 속도 성분
        L_vAxis = L_Vt_inflow_Axis + L_vFree_Axis + L_Vinduced_Axis
        L_vChord = L_Vt_inflow_Chord + L_vFree_Chord + L_Vinduced_Chord
        L_vSpan = L_Vt_inflow_Span + L_vFree_Span + L_Vinduced_Span

        # -------------------------------
        # 아래 루프는 Aeromodel 함수를 통해 무작위 입력에 따른
        # 공력 계수를 계산하지만, 결과를 사용하지 않으므로 최적화를 위해 주석 처리함.
        # -------------------------------

        # for temp in range(30):
        #     thickness = random.uniform(5, 10)
        #     RE = random.uniform(25000, 150000)
        #     AOA = random.uniform(-12, 12)
        #     data = Aeromodel(temp % 3, thickness, RE, AOA)
        #     # print(data)


        # 시각화를 위한 스케일 조정
        scl = 0.001
        L_vAxis = L_vAxis * scl
        L_vChord = L_vChord * scl
        L_vSpan = L_vSpan * scl

        # 블레이드 요소의 위치 (지면 좌표계)
        position = Rot_mat @ (RAzimuth @ np.array([r, 0, 0]))

        # 3D quiver plot으로 각 속도 성분을 시각화
        ax.quiver(position[0], position[1], position[2],
                  L_vAxis[0], L_vAxis[1], L_vAxis[2],
                  color='r')
        ax.quiver(position[0], position[1], position[2],
                  L_vChord[0], L_vChord[1], L_vChord[2],
                  color='g')
        ax.quiver(position[0], position[1], position[2],
                  L_vSpan[0], L_vSpan[1], L_vSpan[2],
                  color='b')

        # 아래 코드는 블레이드 축과 코드, span 벡터의 외적 차이를 출력하는데,
        # 현재는 사용되지 않으므로 주석 처리함.
        # print(np.cross(L_vChord, L_vSpan) / np.linalg.norm(np.cross(L_vChord, L_vSpan)) - BLP_G_unit_axis)

    # azimuth 별로 Figure 2에 scatter plot (속도 성분의 축 방향 투영값)
    plt.figure(2)
    plt.scatter(Now_Azimuth_Angle, np.dot(L_vAxis, BLP_G_unit_axis), color='r')
    plt.scatter(Now_Azimuth_Angle, np.dot(L_vChord, BLP_G_unit_chord_neg), color='g')
    plt.scatter(Now_Azimuth_Angle, np.dot(L_vSpan, BLP_G_unit_span), color='b')

# 3D 플롯의 축 비율을 같게 설정
ax.axis('equal')
print(f"{time.time() - start:.4f} sec")
print()

plt.show()
