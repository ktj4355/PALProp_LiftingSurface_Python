import joblib
import numpy as np
import numpy.linalg as LA

import csv
#import CalcModule  as Calc

from fontTools.misc.bezierTools import epsilon
from numpy.ma.core import zeros_like
from VortexModel import Vortex_Vatistas
from numba import njit
def sind(x):
    return np.sin(np.deg2rad(x))
def cosd(x):
    return np.cos(np.deg2rad(x))
def tand(angle):
    """angle(도)의 탄젠트값 반환"""
    return np.tan(np.deg2rad(angle))
def Rz(deg):
    outMat=np.array([   [cosd(deg)  ,-sind(deg)     , 0],
                        [sind(deg)  , cosd(deg)     , 0],
                        [0          , 0             , 1]])
    return outMat


@njit(cache=True)
def calculate_wake_induced_velocity(WakeMarker, Gamma_TR,Gamma_Shed,coloP, rc):
    """
    Wake가 유도하는 속도를 계산하는 함수 (클래스 외부에서 동작).

    Args:
        WakeMarker (np.ndarray): Wake 마커의 위치 데이터 (NxMx3).
        Gamma_TR (np.ndarray): Trailing wake의 감마 값 (NxM).
        coloP (np.ndarray): 계산할 Colocation 포인트의 위치 (Px3).
        rc (np.ndarray): Colocation 반경.

    Returns:
        np.ndarray: 계산된 유도 속도 벡터 (Px3).
    """
    if WakeMarker.shape[0] < 2:
        return np.zeros_like(coloP)

    Vout = np.zeros_like(coloP)
    for Calc_point_IDX in range(coloP.shape[0]):
        coloP_local = coloP[Calc_point_IDX, :]
        vindTR = np.array([0.0, 0.0, 0.0])
        for TRWake_Row_IDX in range(WakeMarker.shape[0] - 1):
            for TRWake_Col_IDX in range(Gamma_TR.shape[1]):
                colIND = TRWake_Col_IDX * 3
                colIND2 = (TRWake_Col_IDX+1) * 3
                WakeMarker_A = WakeMarker[TRWake_Row_IDX, colIND: colIND + 3]
                WakeMarker_B = WakeMarker[TRWake_Row_IDX + 1, colIND: colIND + 3]
                vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, coloP_local,Gamma_TR[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX],1.06)
                if TRWake_Col_IDX==Gamma_TR.shape[1]-1:continue
                WakeMarker_A=WakeMarker[TRWake_Row_IDX, colIND: colIND + 3]
                WakeMarker_B=WakeMarker[TRWake_Row_IDX, colIND2: colIND2 + 3]
                vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, coloP_local,
                                        Gamma_Shed[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX],1.06)
        Vout[Calc_point_IDX, :] = vindTR
    return Vout

@njit(cache=True)
def calculate_wake_velocity(WakeMarker, Gamma_TR,Gamma_Shed,Gamma_BD,BDpoint,TEPoint, rc,rcBD):

    if WakeMarker.shape[0] < 2:
        return np.zeros_like(WakeMarker)

    Vout = np.zeros_like(WakeMarker)
    for marker_wake in range(WakeMarker.shape[0]):
        for marker_geom in range(Gamma_TR.shape[1]):
            marker_geom_ind=marker_geom*3
            Marker_local = WakeMarker[marker_wake, marker_geom_ind:marker_geom_ind+3]
            vindTR = np.array([0.0, 0.0, 0.0])
            #print("Wake Calc")
            #print(rcBD)
            for TRWake_Row_IDX in range(WakeMarker.shape[0] - 1):
                for TRWake_Col_IDX in range(Gamma_TR.shape[1]):
                    colIND = TRWake_Col_IDX * 3
                    colIND2 = (TRWake_Col_IDX+1) * 3
                    #Trailing Wake
                    #print("trWake")
                    WakeMarker_A = WakeMarker[TRWake_Row_IDX, colIND: colIND + 3]
                    WakeMarker_B = WakeMarker[TRWake_Row_IDX + 1, colIND: colIND + 3]


                    vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,Gamma_TR[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX],1.06)
                    # Shed Wake


                    if TRWake_Col_IDX==Gamma_TR.shape[1]-1:continue
                    WakeMarker_A=WakeMarker[TRWake_Row_IDX, colIND: colIND + 3]
                    WakeMarker_B=WakeMarker[TRWake_Row_IDX, colIND2: colIND2 + 3]
                    vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                            Gamma_Shed[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX],1.06)


             # Blade Induced

            #print("BD Calc")
            for BDWake_Row_IDX in range(Gamma_BD.shape[0]):
                WakeMarker_A=BDpoint[BDWake_Row_IDX,:]
                WakeMarker_B=BDpoint[BDWake_Row_IDX+1,:]
                '''
                print(WakeMarker_A)
                print(WakeMarker_B)
                print(Marker_local)
                print(Gamma_BD[BDWake_Row_IDX])
                print(rcBD[BDWake_Row_IDX])
                print(Vortex_Scully(WakeMarker_A, WakeMarker_B, Marker_local,Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX]))
                '''
                vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX],1.06)

                WakeMarker_A = TEPoint[BDWake_Row_IDX, :]
                WakeMarker_B = BDpoint[BDWake_Row_IDX, :]
                vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                        Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX],1.06)
                WakeMarker_A = BDpoint[BDWake_Row_IDX + 1, :]
                WakeMarker_B = TEPoint[BDWake_Row_IDX + 1, :]
                vindTR += Vortex_Vatistas(WakeMarker_A, WakeMarker_B, Marker_local,
                                        Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX],1.06)

            Vout[marker_wake, marker_geom_ind:marker_geom_ind+3] = vindTR

    return Vout



class Rotor:

    def __init__(self):
        # initial Value
        self.inputGeom=np.array([[0,0,0,0,0]])
        self.PannelGeom = self.inputGeom
        # rotor Geometric Value
        self.R=0
        self.bmean=1
        self.R_hub=0.084
            # 틸트 세팅


        # Calculation Setup
        self.alt = 0 #m
        self.T, self.P, self.den, self.D_vis, self.a = self.stdAtm(self.alt)

        self.vFree = [0, 0, -0] #FreeVelocity
        if self.vFree[0]==0 :
            self.vFree[0] == 0.00001
        elif self.vFree[1]==0:
            self.vFree[1] == 0.00001
        elif self.vFree[2]==0:
            self.vFree[2] == 0.00001
        self.nAzmuth = 18 #slice of rotation
        self.Tilt_angle=0
        self.Tilt_Phi=0

        self.dAngle = 0
        self.RPM = 5000 #rev / min
        self.dt=0
        self.Blade = 2

        pitch = np.deg2rad(self.Tilt_angle)
        Rp = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # z축 회전 (요 회전)
        yaw = np.deg2rad(self.Tilt_Phi)
        Ry = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # 두 회전을 결합하여 프로펠러의 전체 회전행렬 생성
        self.Rot_mat = Ry @ Rp

        # AerodynamicSetting
        self.BlendPosition = np.array([2.85, 4.65]) * 0.0254
        self.Calc_Setting()
        #self.[Tmp, Pressure, rho, D_vis, a] = STD_Atm(alt)
        self.warmup()
    def warmup(self):
        # 테스트용 초기값 설정
        A = np.array([1.0, 0.0, 0.0])
        B = np.array([0.0, 1.0, 0.0])
        ColocationPoint = np.array([0.5, 0.5, 0.0])
        vortexStrength = 1.0
        rc = 0.1

        # 함수 호출로 컴파일 강제 수행
        _ = Vortex_Vatistas(A, B, ColocationPoint, vortexStrength, rc,1.06)
        self.Calc_Wake_Induced_Velocity_XYZ()
    def Calc_Setting(self):
        self.ZeroGeom = np.array([0, 0, 2 * self.R_hub, 0, 0.008])
        self.AR = 2 * (self.R) / self.bmean
        self.D = self.R * 2  # m
        self.n = self.RPM / 60
        # J = V. / (n. * D);
        self.V_tip = 2 * np.pi * self.n * self.R
        self.M_tip = self.V_tip / 340
        self.Afan = np.pi * self.R * self.R
        # Blended Position Should Be converted to Non Dimension
        self.BlendPosition = self.BlendPosition / self.R
        self.BladeChord = self.inputGeom[:,2]

        # initial colocationFactor
        self.PannelGeom_r = np.zeros([self.inputGeom.shape[0] - 1, 2])
        self.PannelGeom_ds = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom_dc = np.zeros([self.inputGeom.shape[0] - 1, 1])

        self.PannelGeom_Chorld = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom_Beta = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom_Thickness = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom =np.zeros([self.inputGeom.shape[0] - 1, 5])
        self.collocation_point_local = np.zeros([self.inputGeom.shape[0] - 1, 2])

        self.Calc_PanelGeom()

        # initial Bound Gamma Vector
        self.Gamma_bound = np.ones([self.inputGeom.shape[0]-1, 1])
        #self.now_Gamma_bound = np.ones([self.inputGeom.shape[0]-1, 1])

        # Vortex Core Radius
        self.rc_panel=(self.PannelGeom_Chorld*0.1).copy()
        self.rc_Geom=(self.BladeChord*0.1).copy()
        print(self.rc_panel)
        print(self.rc_Geom)
        # Current Position Initializing
            #Prop Geometry
        self.now_Prop_position_Bound= np.zeros([self.inputGeom.shape[0], 3]) #x y z coord
        self.now_Prop_position_LE= np.zeros([self.inputGeom.shape[0], 3]) #x y z coord
        self.now_Prop_position_TE= np.zeros([self.inputGeom.shape[0], 3]) #x y z coord
        self.now_Angle=0
            # Panel Geometry
        self.now_Panl_position_Bound = np.zeros([self.inputGeom.shape[0], 3])  # x y z coord
        self.now_Panl_position_LE = np.zeros([self.inputGeom.shape[0], 3])  # x y z coord
        self.now_Panl_position_TE = np.zeros([self.inputGeom.shape[0], 3])  # x y z coord
            # Colocation Geometry
        self.now_Colo_position = np.zeros([self.inputGeom.shape[0], 3])  # x y z coord
        self.Calc_Blade_Rotation_Position_XYZ(0)
            # Trailing Wake Vortex
        self.WakeMarker_Position = self.convertWakeGeom(self.now_Prop_position_TE).copy()  # x y z coord
        self.Gamma_Wake_TR = np.zeros([1,self.inputGeom.shape[0]])  # Trailing Wake strengh
        self.TE_induced_Vel=np.zeros_like(self.WakeMarker_Position)

        self.Gamma_Wake_Shed = np.zeros([1,self.inputGeom.shape[0]-1])  # Shed Wake Strength
        self.WakeMarker_Vel=np.zeros_like(self.WakeMarker_Position)
            # Shed Wake Vortex

            # Disk Induced Velocity
        self.Vel_colocation_Bound_Induced_XYZ=np.zeros([self.inputGeom.shape[0]-1,3])
        self.Vel_colocation_Wake_Induced_XYZ = np.zeros([self.inputGeom.shape[0] - 1, 3])
        self.Vel_colocation_Total_XYZ = np.zeros([self.inputGeom.shape[0] - 1, 3])

            # Blade FLow Velocity
        self.Phi_flow = 0
        self.vA = 0
        self.vT = 0

            # 틸트 세팅
        pitch = np.deg2rad(self.Tilt_angle)
        Rp = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

            # z축 회전 (요 회전)
        yaw = np.deg2rad(self.Tilt_Phi)
        Ry = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
            # 두 회전을 결합하여 프로펠러의 전체 회전행렬 생성
        self.Rot_mat = Ry @ Rp
        groundUnit_X = np.array([1, 0, 0])
        groundUnit_Y = np.array([0, 1, 0])
        groundUnit_Z = np.array([0, 0, 1])

        # ===============================
        # Disk (프로펠러) 프레임 설정
        # ===============================
        # 프로펠러의 회전행렬을 이용하여 disk 프레임의 단위벡터를 정의
        self.Unit_Disk_tanX = self.Rot_mat @ groundUnit_X  # 디스크 접선 방향 (azimuth 0)
        self.Unit_Disk_tanY = self.Rot_mat @ groundUnit_Y  # 디스크 접선 방향 (azimuth 90)
        self.Unit_Disk_Axis = self.Rot_mat @ groundUnit_Z  # 디스크의 법선 (축)

    def convertWakeGeom(self,geom):
        wakeGeom=geom.reshape(3*geom.shape[0])
        return np.array([wakeGeom])
    def Calc_PanelGeom(self):
        # Pannel Geom
        # prop Geom IDX  1---2---3---4---5---6---7---8
        # Panel Geom IDX |-1-|-2-|-3-|-4-|-5-|-6-|-7-|
        # Coloation Geom IDX is Same to Panel Geom
        # r/R r Chord Beta(deg) TicknessRatio


        for idx in range(self.inputGeom.shape[0]-1):
            Section1 = self.inputGeom[idx,:].copy()
            Section2 = self.inputGeom[idx + 1,:].copy()
            self.PannelGeom[idx,:]=((Section1+Section2)/2).copy()
            #print(idx)
            #print(PannelGeom_tmp)
            #print()

            #PannelGeom : 각 패널의 중앙위치
            self.PannelGeom_r[idx,:]=np.array([self.PannelGeom[idx,0], self.PannelGeom[idx,1]]).copy()
            self.PannelGeom_Chorld[idx]= self.PannelGeom[idx,2].copy()
            self.PannelGeom_Beta[idx] = self.PannelGeom[idx,3].copy()
            self.PannelGeom_Thickness[idx] = self.PannelGeom[idx,4].copy()

            self.PannelGeom_dc[idx]=abs(self.inputGeom[idx+1,1]-self.inputGeom[idx,1]).copy()
            self.PannelGeom_ds[idx]=(abs(self.inputGeom[idx+1,1]-self.inputGeom[idx,1])*self.PannelGeom[idx,2]).copy()

            # Twist Center is 0.25c
            # Colocation Point is 0.75c
            # LE----BD----/-----Col------TE
            # 0.25c -----0 ------0.75c----Te

            #Calculate Real Position
            #Geometry Bound Vortex Point, Leading Edge Point, TailEdge point
            #Coordinate : [Radial, Normal, Tangential]
            #기준점 : 0.25C
        r =  self.inputGeom[:,1] #local section radial position
        n = (self.inputGeom[:, 2]*0.25)*sind(self.inputGeom[:, 3]).copy()
        t = (self.inputGeom[:, 2]*0.25)*cosd(self.inputGeom[:, 3]).copy()
        self.GeomPointVector_BD_Global_ini = np.c_[r, r*0, r*0].copy() #[r, 0, 0]
        self.GeomPointVector_LE_Global_ini = np.c_[r, t, n].copy()
        self.GeomPointVector_TE_Global_ini = np.c_[r, -3 * t, -4 * n].copy()


        #Colocation
        r =  self.PannelGeom[:,1] #local section radial position
        n = (self.PannelGeom[:, 2]*0.25)*sind(self.PannelGeom[:, 3]).copy()
        t = (self.PannelGeom[:, 2]*0.25)*cosd(self.PannelGeom[:, 3]).copy()
        self.collocation_point_Global_ini = np.c_[r, -t*2, -n].copy() #[r, 0, 0]

        self.PanlPointVector_BD_Global_ini = np.c_[r, r * 0, r * 0].copy()  # [r, 0, 0]
        self.PanlPointVector_LE_Global_ini = np.c_[r, t, n].copy()
        self.PanlPointVector_TE_Global_ini = np.c_[r, -3 * t, -4 * n].copy()
       # self.collocation_point_local[idx,:]=np.array([self.PannelGeom[idx,1],-self.PannelGeom[idx,2]*0.75,0])
        #print(self.PannelGeom)
    def readImportFile(self, fPath):
        numeric_data = []

        try:
            self.ReadAerodynamicModel()
            csv_file=open(fPath, 'r')
        except:
            numeric_data = [0, 0, 0, 0, 0]
            return np.array([[0, 0, 0, 0, 0]])
        reader = csv.reader(csv_file)

            # 첫 줄(헤더) 건너뛰기
        next(reader, None)
        #print(reader)
        for row in reader:
            numeric_row = []
            for item in row:
                try:
                    # 숫자로 변환
                    numeric_row.append(float(item))
                except ValueError:
                    # 변환 실패 시 0 추가
                    numeric_row.append(0.0)
            numeric_data.append(numeric_row)
        csv_file.close()
        self.inputGeom=np.array(numeric_data)
        self.R = self.inputGeom[self.inputGeom.shape[0] - 1, 1]
        self.bmean = np.median(self.inputGeom[:, 2])
        self.Calc_Setting()
        return np.array(numeric_data)
    def Calc_Iij_BoundMatrix(self):
        CollocPoint=self.now_Colo_position.copy()


        Prop_position_Bound= self.now_Prop_position_Bound
        Prop_position_LE= self.now_Prop_position_LE
        Prop_position_TE= self.now_Prop_position_TE

        Panl_position_Bound= self.now_Panl_position_Bound
        Panl_position_LE= self.now_Panl_position_LE

        Colo_position= self.now_Colo_position
        #induced Velocity Temp Array
        gamma_ONES = np.ones([self.Gamma_bound.shape[0], 1])
        vind_Bound = np.zeros([gamma_ONES.shape[0], 3])
        vind_trLeft = np.zeros([gamma_ONES.shape[0], 3])
        vind_trRight = np.zeros([gamma_ONES.shape[0], 3])
        vind =np.zeros([gamma_ONES.shape[0],3])

        #I Matrix
        self.now_I_ini_x = np.zeros([Colo_position.shape[0],gamma_ONES.shape[0]])
        self.now_I_ini_y = np.zeros([Colo_position.shape[0],gamma_ONES.shape[0]])
        self.now_I_ini_z = np.zeros([Colo_position.shape[0],gamma_ONES.shape[0]])
        for colocationInd in range(Colo_position.shape[0]):

            ColoP = Colo_position[colocationInd, :]
            for gamma_idx in range(gamma_ONES.shape[0]):
                gamma=gamma_ONES[gamma_idx]
                #print(Prop_position_Bound[gamma_idx,:])
                #print(Prop_position_TE[gamma_idx+1,:])
                #print(ColoP)
                #print(rc[gamma_idx])
                vind_Bound[gamma_idx,:]=Vortex_Vatistas(Prop_position_Bound[gamma_idx,:],Prop_position_Bound[gamma_idx+1,:],ColoP,gamma, self.rc_panel[gamma_idx],1.06)
                vind_trLeft[gamma_idx,:]=Vortex_Vatistas(Prop_position_TE[gamma_idx,:],Prop_position_Bound[gamma_idx,:],ColoP,gamma, self.rc_panel[gamma_idx],1.06)
                vind_trRight[gamma_idx,:]= Vortex_Vatistas(Prop_position_Bound[gamma_idx+1,:],Prop_position_TE[gamma_idx+1,:],ColoP,gamma, self.rc_panel[gamma_idx],1.06)
                vind[gamma_idx,:]=(vind_Bound[gamma_idx,:]+vind_trLeft[gamma_idx,:]+vind_trRight[gamma_idx,:]).copy()
                self.now_I_ini_x[colocationInd, gamma_idx]    = vind[gamma_idx,0].copy()
                self.now_I_ini_y[colocationInd, gamma_idx]   = vind[gamma_idx,1].copy()
                self.now_I_ini_z[colocationInd, gamma_idx]   = vind[gamma_idx,2].copy()
    def Calc_Blade_Rotation_Position_XYZ(self,Angle_deg):
        # Current Position Initializing
        # Prop Geometry
        self.dAngle=Angle_deg-self.now_Angle
        if self.dAngle<0:
            self.dAngle= self.dAngle+360
        self.now_Angle=Angle_deg
        self.now_Prop_position_Bound = (Rz(Angle_deg)@self.GeomPointVector_BD_Global_ini.T).T.copy()  # x y z coord
        self.now_Prop_position_LE = (Rz(Angle_deg)@self.GeomPointVector_LE_Global_ini.T).T.copy()  # x y z coord
        self.now_Prop_position_TE = (Rz(Angle_deg)@self.GeomPointVector_TE_Global_ini.T).T.copy()  # x y z coord

        # Panel Geometry
        self.now_Panl_position_Bound =(Rz(Angle_deg)@self.PanlPointVector_BD_Global_ini.T).T.copy()  # x y z coord
        self.now_Panl_position_LE = (Rz(Angle_deg) @ self.PanlPointVector_LE_Global_ini.T).T.copy()
        self.now_Panl_position_TE = (Rz(Angle_deg) @ self.PanlPointVector_TE_Global_ini.T).T.copy()

        # Colocation Geometry
        self.now_Colo_position = (Rz(Angle_deg)@self.collocation_point_Global_ini.T).T.copy()   # x y z coord
        # Azimuth Vector Setting
        # 현재 방위각에 따른 회전행렬 생성 (도 단위 사용)
        RAzimuth = np.array([
            [cosd(Angle_deg), -sind(Angle_deg), 0],
            [sind(Angle_deg), cosd(Angle_deg), 0],
            [0, 0, 1]
        ])

        BLP_unit_chord_neg = RAzimuth @ np.array([0, 1, 0])
        BLP_unit_span = RAzimuth @ np.array([1, 0, 0])
        BLP_unit_axis = RAzimuth @ np.array([0, 0, 1])

        self.BLP_G_unit_tan = self.Rot_mat @ BLP_unit_chord_neg
        self.BLP_G_unit_span = self.Rot_mat @ BLP_unit_span
        self.BLP_G_unit_axis = self.Rot_mat @ BLP_unit_axis
    def Calc_Bound_Induced_Velocity_XYZ(self):
        #방위각에 따라 계산된 Imat에 BoundGammaMatrix를 곱하여 유도속도 계산
        #Blade에 속박되어있는 와류만 계산
        I_ini_x=self.now_I_ini_x.copy()
        I_ini_y=self.now_I_ini_y.copy()
        I_ini_z=self.now_I_ini_z.copy()
        vx=I_ini_x@self.Gamma_bound
        vy=I_ini_y@self.Gamma_bound
        vz=I_ini_z@self.Gamma_bound
        self.Vel_colocation_Bound_Induced_XYZ=np.c_[vx,vy,vz].copy()
    def Calc_Wake_Induced_Velocity_XYZ(self):
        WakeMarker = self.WakeMarker_Position.copy()
        Gamma_TR = self.Gamma_Wake_TR.copy()
        Gamma_Shed = self.Gamma_Wake_Shed.copy()
        coloP = self.now_Colo_position.copy()
        rc = self.rc_Geom.copy()
        #print("marker",WakeMarker)
        #print("tr",Gamma_TR)
        #print("shed",Gamma_Shed)
        #print("colop",coloP)
        #print("rc",rc)

        # 외부 함수 호출
        self.Vel_colocation_Wake_Induced_XYZ = calculate_wake_induced_velocity(WakeMarker, Gamma_TR,Gamma_Shed, coloP, rc)
        #print("done")
    def Calc_Wake_Velocity(self):
        WakeMarker = self.WakeMarker_Position.copy()
        Gamma_TR = self.Gamma_Wake_TR.copy()
        Gamma_Shed = self.Gamma_Wake_Shed.copy()
        rc = self.rc_Geom.copy()
        Gamma_BD=self.Gamma_bound
        BDpoint=self.now_Prop_position_Bound
        TEPoint=self.now_Prop_position_TE
        rcBD=self.rc_panel.reshape(-1).copy()


        # 외부 함수 호출

        #print("ddddone")
        '''
        print("WakeMarker : ", WakeMarker)
        print("Gamma_TR : ", Gamma_TR)
        print("Gamma_Shed : ", Gamma_Shed)
        print("Gamma_BD : ", Gamma_BD)

        print("BDpoint : ", BDpoint)
        print("TEPoint : ", TEPoint)
        print("rc : ", rc)
        print("rcBD : ", rcBD)
        '''
        self.WakeMarker_Vel = calculate_wake_velocity(WakeMarker, Gamma_TR,Gamma_Shed,Gamma_BD,BDpoint,TEPoint, rc,rcBD)

        #self.history_TEWake=self.WakeMarker_Vel[0,:].copy()
        #self.WakeMarker_Vel[0, :]=0

    def WakeUpdate(self):
        #Wake의 Timestep Update
        #VLM이 다 게산되고, Updated된 Geom에 맞추어 확정된 Gamma를 이용하여 각 Point에 유도되는 속도계산
        #오일러 1차 양해법으로 앞으로 전진
        # 유도속도, Wake 유도속도에 의해 각 Wake Point의 속도 계산
        # V= FreeVel + Colocation Point Induced Vel + InducedVel의 Rotation Speed (Propeller Vortex), 각 Wake에의해 유도된속도
        #
        #rint(self.history_TEWake)

        #self.TE_induced_Vel = np.vstack((self.history_TEWake, self.TE_induced_Vel.copy())).copy()
        #self.TE_induced_Vel[-1, :] = self.TE_induced_Vel[-2, :].copy()
        #TE_induced_History= self.TE_induced_Vel[:-2,:].copy()

        boundGamma = np.transpose(self.Gamma_bound.copy())
        oldMarker = self.WakeMarker_Position.copy()
        freeVel=np.ones_like(oldMarker)
        self.dt=60*(self.dAngle/(self.RPM*360.0))
        for ind in range(boundGamma.shape[1]+1):
            freeVel[:,ind*3]=(freeVel[:,ind*3]*self.vFree[0]).copy()
            freeVel[:,ind*3+1]=(freeVel[:,ind*3+1]*self.vFree[1]).copy()
            freeVel[:,ind*3+2]=(freeVel[:,ind*3+2]*self.vFree[2]).copy()
        #print("TE Induce")
        #print(self.TE_induced_Vel)
        #print()
       # oldMarker=oldMarker+(freeVel+TE_induced_History+self.WakeMarker_Vel)*self.dt
        oldMarker = oldMarker + (freeVel + self.WakeMarker_Vel) * self.dt
       # print(freeVel*self.dt)

        # 각 Wake Point에 유도되는 속도 계산 코드

        # 기존 Wake Geomerty의 위치 업데이트 dr=Vdt

        #밖으로 떨어져나오는 Wake의 Vortex 계산

        #oldTrailingWake_Gamma=


        newTrailingWake_gamma=np.zeros([1,boundGamma.shape[1]+1])
        newTrailingWake_gamma[0,0]=-boundGamma[0,0]
        tmp=0
        for ind in range(boundGamma.shape[1]-1):
            tmp=ind
            newTrailingWake_gamma[0, ind+1] =boundGamma[0,ind]-boundGamma[0,ind+1]
        newTrailingWake_gamma[0,tmp+2]=boundGamma[0,tmp+1]

        shedVortex=np.array([newTrailingWake_gamma[0,:]-self.Gamma_Wake_TR[0,:]]).copy()
        endIDX=shedVortex.shape[1]-1
        shedVortex=shedVortex[:,0:endIDX].copy()

        #oldShedWake_gamma
        #newShedWake_gamma



        newWakeMarker=self.convertWakeGeom(self.now_Prop_position_TE).copy()

        #Wake Matrix Update

        self.WakeMarker_Position=np.concatenate((newWakeMarker,oldMarker),axis=0).copy()
        self.Gamma_Wake_TR=np.concatenate((newTrailingWake_gamma,self.Gamma_Wake_TR),axis=0).copy()
        self.Gamma_Wake_Shed=np.concatenate((shedVortex,self.Gamma_Wake_Shed),axis=0).copy()
    def VLM_GammaUpdate(self):
        # Colocation Point의 유도속도를 기반으로해서 Inflow 속도 계산
        # 좌표계 변환을 통해 각 익형의 받음각, Cl Cd 계산
        # 각 익형 Section의 힘으로 변환한 후 힘 ,토크 계산
        # 계산된 Cl Cd를 이용하여 BoundVortex 새로 계산

        pass

    def Timemarching(self):
        # VLM 계산
        # Wake Update
        # Geom Update
        # 힘 정보 업데이트
        # Dt 증가 및 업데이트


        pass
    def ReadAerodynamicModel(self):

        # ===============================
        # Aerodynamic Model Initialization
        # ===============================
        # 각 구간(Section1, Section2, Blade)의 양력/항력 계수 모델을 joblib를 이용하여 불러옴
        print("read ...")
        self.Section1_CL = joblib.load('Section1_CL.pkl')
        self.Section1_CD = joblib.load('Section1_CD.pkl')
        self.Section1_CL_near = joblib.load('Section1_CL_near.pkl')
        self.Section1_CD_near = joblib.load('Section1_CD_near.pkl')
        # 모델의 초기 호출(예: 캐싱이나 확인 목적)
        self.Section1_CL(0, 0, 0)
        self.Section1_CD(0, 0, 0)
        self.Section1_CL_near(0, 0, 0)
        self.Section1_CD_near(0, 0, 0)

        self.Section2_CL = joblib.load('Section2_CL.pkl')
        self.Section2_CD = joblib.load('Section2_CD.pkl')
        self.Section2_CL_near = joblib.load('Section2_CL_near.pkl')
        self.Section2_CD_near = joblib.load('Section2_CD_near.pkl')
        self.Section2_CL(0, 0, 0)
        self.Section2_CD(0, 0, 0)
        self.Section2_CL_near(0, 0, 0)
        self.Section2_CD_near(0, 0, 0)

        self.BLD_CL = joblib.load('BLD_CL.pkl')
        self.BLD_CD = joblib.load('BLD_CD.pkl')
        self.BLD_CL_near = joblib.load('BLD_CL_near.pkl')
        self.BLD_CD_near = joblib.load('BLD_CD_near.pkl')
        self.BLD_CL(0, 0, 0)
        self.BLD_CD(0, 0, 0)
        self.BLD_CL_near(0, 0, 0)
        self.BLD_CD_near(0, 0, 0)
        print("Done!")
    #@njit(cache=True)
    def Aeromodel(self, model, arg1, arg2, arg3):

        # ===============================
        # Aeromodel 함수: 주어진 입력값에 따라 적절한 모델을 선택하여
        # 양력(cl)과 항력(cd)를 계산함
        # ===============================
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
            cl =  self.Section1_CL(arg1, arg2, arg3)
            cd =  self.Section1_CD(arg1, arg2, arg3)
            # cl 값이 매우 클 경우 near-stall 모델 사용
            if cl > 100:
                cl =  self.Section1_CL_near(arg1, arg2, arg3)
                cd =  self.Section1_CD_near(arg1, arg2, arg3)
            return np.array([cl, cd])

        if model == 2:
            cl =  self.Section2_CL(arg1, arg2, arg3)
            cd =  self.Section2_CD(arg1, arg2, arg3)
            if cl > 100:
                cl =  self.Section2_CL_near(arg1, arg2, arg3)
                cd =  self.Section2_CD_near(arg1, arg2, arg3)
            return np.array([cl, cd])
        else:
            cl =  self.BLD_CL(arg1, arg2, arg3)
            cd =  self.BLD_CD(arg1, arg2, arg3)
            if cl > 100:
                cl =  self.BLD_CL_near(arg1, arg2, arg3)
                cd =  self.BLD_CD_near(arg1, arg2, arg3)
            return np.array([cl, cd])
    def RadiusVectorSet(self, rR,rpm, G_Inflow_Vehicle_vec, G_v_ind_col):
        r = self.R * rR  # 실제 블레이드의 반경 위치
        #print(rpm)
        #print(r)

        # 해당 반경에서의 회전에 의한 접선 속도벡터 계산
        Vt_rot = (2 * np.pi * (rpm/60.0) * r) * self.BLP_G_unit_tan

        # 회전, 자유 유동, 유도 속도의 각 성분 계산
        L_Vt_Axis = 0 * self.BLP_G_unit_axis  # 축 방향은 0
        L_Vt_tan = - Vt_rot  # 코드 방향(항력 관련)
        L_Vt_Span = 0 * self.BLP_G_unit_span  # span 방향은 0

        # 자유 유동(차량 속도)의 각 성분 (지면 기준)
        L_vFree_Axis = np.dot(G_Inflow_Vehicle_vec, self.BLP_G_unit_axis) * self.BLP_G_unit_axis
        L_vFree_tan = np.dot(G_Inflow_Vehicle_vec, self.BLP_G_unit_tan) * self.BLP_G_unit_tan
        L_vFree_Span = np.dot(G_Inflow_Vehicle_vec, self.BLP_G_unit_span) * self.BLP_G_unit_span

        # 유도 속도의 각 성분 (지면 기준)
        L_Vinduced_Axis = np.dot(G_v_ind_col, self.BLP_G_unit_axis) * self.BLP_G_unit_axis
        L_Vinduced_tan = np.dot(G_v_ind_col, self.BLP_G_unit_tan) * self.BLP_G_unit_tan
        L_Vinduced_Span = np.dot(G_v_ind_col, self.BLP_G_unit_span) * self.BLP_G_unit_span

        # 각 성분의 합산: 블레이드 요소에서의 전체 유동 속도 성분 (지면기준)
        L_vAxis = L_Vt_Axis + L_vFree_Axis + L_Vinduced_Axis
        L_vtan = L_Vt_tan + L_vFree_tan + L_Vinduced_tan
        L_vSpan = L_Vt_Span + L_vFree_Span + L_Vinduced_Span

        self.W_flow = np.linalg.norm(L_vAxis + L_vtan)
        self.Phi_flow = np.rad2deg(np.atan2(-np.dot(L_vAxis, self.BLP_G_unit_axis), -np.dot(L_vtan, self.BLP_G_unit_tan)))
        self.vT = np.dot(L_vtan, self.BLP_G_unit_tan)
        self.vA = np.dot(L_vAxis, self.BLP_G_unit_axis)

        # print(Phi_flow, W_flow)


    def VLM(self,Angle_deg):
        # Bound Gamma 추정
        # Induced Velocity 계산
        key=0
        self.Calc_Blade_Rotation_Position_XYZ(Angle_deg)
        self.Calc_Iij_BoundMatrix()

        # 다음은 Pannel Geometry에 대한 변수임
        r=self.PannelGeom_r[:,1].copy()
        chord=self.PannelGeom_Chorld.copy()
        beta=self.PannelGeom_Beta.copy()
        thick=self.PannelGeom_Thickness.copy()
        #self.collocation_point_local
        ds=self.PannelGeom_ds.copy()
        R=self.R
        rR=self.PannelGeom_r[:,0].copy()

       # T,Q,P,Fx,Fy,Fz=(0, 0, 0, 0, 0, 0)
        oldT,oldQ,oldP,oldFx,oldFy,oldFz=(999, 999, 999, 999, 999, 999)
        self.Calc_Bound_Induced_Velocity_XYZ()
        self.Calc_Wake_Induced_Velocity_XYZ()
        self.Vel_colocation_Total_XYZ = self.Vel_colocation_Bound_Induced_XYZ + self.Vel_colocation_Wake_Induced_XYZ

        # 현재 위치에 대해 rR별로 계산
        iter=0
        regvel=0
        Old_regVel=999
        while iter<500:
            iter=iter+1

            T, Q, P, F,Fx, Fy, Fz = (0,0, 0, 0, 0, 0, 0)
            self.Calc_Bound_Induced_Velocity_XYZ()
            self.Calc_Wake_Induced_Velocity_XYZ()
            self.Vel_colocation_Total_XYZ = self.Vel_colocation_Bound_Induced_XYZ + self.Vel_colocation_Wake_Induced_XYZ

            for r_idx in range(rR.shape[0]):

                G_Inflow_Free_vec = self.vFree
                G_v_ind_col=self.Vel_colocation_Total_XYZ[r_idx,:].copy()
                #print(G_v_ind_col)
                #print(G_v_ind_col)
                self.RadiusVectorSet(rR[r_idx], self.RPM, G_Inflow_Free_vec, G_v_ind_col)
                FlowAngle=self.Phi_flow
                V_effective=self.W_flow
                vA=self.vA
                vT=self.vT
                Re=self.den*V_effective*chord[r_idx]/self.D_vis
                AoA=beta[r_idx]-FlowAngle
                #print(thick[r_idx],Re,AoA,ds[r_idx])
                aero=self.Aeromodel(1,thick[r_idx],Re,AoA)
                Cl=aero[0]
                Cd=aero[1]
                #print(f"veff at {rR[r_idx]:.3f} {vA}")

                LD=Cl/Cd
                rLoc=r[r_idx]
                b=chord[r_idx]
                Gam=np.rad2deg(np.atan(1 / LD)) #atan(D/L)
                Ks=0.5*self.den*(vA**2)*Cl/((sind(FlowAngle)**2)*cosd(Gam))

                #print(Gam,FlowAngle,(sind(FlowAngle)**2)*cosd(Gam))
                Tds=Ks*cosd(Gam+FlowAngle)*ds[r_idx]
                Qds=rLoc*Ks*sind(Gam+FlowAngle)*ds[r_idx]
                Fds=-Ks*sind(Gam+FlowAngle)*ds[r_idx]

                #print(Cl,Cd,vA,self.den,Tds,Qds,Fds)
                T=T+Tds
                Q=Q+Qds
                F=F+Fds
               # print(V_effective)
                Boundgamma=b*V_effective*Cl*0.5

                #Boundgamma=0.1

                if 0<r_idx<rR.shape[0]-1:
                    Boundgamma=( self.Gamma_bound[r_idx-1]+ self.Gamma_bound[r_idx+1]+Boundgamma)/3
                self.Gamma_bound[r_idx]=0.5*Boundgamma+0.5*self.Gamma_bound[r_idx].copy()

                #self.Gamma_bound[r_idx]=0.1

                #dT=cl
                """
                    Parameters:
                      - model: 1, 2, 또는 그 외의 값에 따라 서로 다른 모델을 선택
                      - arg1: 두께(thickness) 등 물리적 파라미터
                      - arg2: Reynolds number (RE)
                      - arg3: Angle of Attack (AOA)
    
                    Returns:
                      - np.array([cl, cd]): 계산된 양력 및 항력 계수
                """

            Tvec=T*self.BLP_G_unit_axis
            Fvec = -F * self.BLP_G_unit_tan
            TotalForce_vec=Tvec+Fvec


            regVel=np.average(self.Gamma_bound)
            if abs(T-oldT)<0.001:
                #print("수렴했습니다 :", Angle_deg,T)
                #print(self.Gamma_bound)

                key=2
                break
            else:
                oldT=T
                Old_regVel=regVel
        print(T, np.average(self.Gamma_bound), AoA)
        return key


        # 추력 계산
        # Bound Gamma 계산
        # Old값과 New값이 수렴할때까지 반복 계산


    def stdAtm(self,Height):
        if Height < 11000:
            T = 288.15 - 0.0065 * Height
            P = 101.325 * (T / 288.15) ** 5.2559
        elif Height < 25000:
            T = 273.15 - 56.46  # 약 216.69 K
            P = 22.65 * np.exp(1.73 - 0.000157 * Height)
        else:  # Height >= 25000
            T = 273.15 - 131.21 + 0.00299 * Height
            P = 2.488 * (T / 216.15) ** (-11.388)

        den = P / (0.2869 * T)
        D_vis = (0.000001458 * T ** 1.5) / (T + 110.4)
        a = np.sqrt(1.4 * 287 * T)

        return T, P, den, D_vis, a

    def clearLastWake(self,n):
        self.WakeMarker_Position = self.WakeMarker_Position[:-n,:].copy()
        self.Gamma_Wake_TR = self.Gamma_Wake_TR[:-n,:].copy()
        self.Gamma_Wake_Shed = self.Gamma_Wake_Shed[:-n,:].copy()
        self.TE_induced_Vel = self.TE_induced_Vel[:-n, :].copy()
