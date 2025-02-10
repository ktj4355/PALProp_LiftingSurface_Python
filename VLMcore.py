import joblib
import numpy as np
import csv

from numpy.ma.core import zeros_like

from VortexModel import Vortex_Scully
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
                vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, coloP_local,
                                        Gamma_TR[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX])
                if TRWake_Col_IDX==Gamma_TR.shape[1]-1:continue
                WakeMarker_A=WakeMarker[TRWake_Row_IDX, colIND: colIND + 3]
                WakeMarker_B=WakeMarker[TRWake_Row_IDX, colIND2: colIND2 + 3]
                vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, coloP_local,
                                        Gamma_Shed[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX])
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


                    vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, Marker_local,Gamma_TR[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX])
                    # Shed Wake


                    if TRWake_Col_IDX==Gamma_TR.shape[1]-1:continue
                    WakeMarker_A=WakeMarker[TRWake_Row_IDX, colIND: colIND + 3]
                    WakeMarker_B=WakeMarker[TRWake_Row_IDX, colIND2: colIND2 + 3]
                    vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, Marker_local,
                                            Gamma_Shed[TRWake_Row_IDX, TRWake_Col_IDX], rc[TRWake_Col_IDX])


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
                vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, Marker_local,Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX])

                WakeMarker_A = TEPoint[BDWake_Row_IDX, :]
                WakeMarker_B = BDpoint[BDWake_Row_IDX, :]
                vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, Marker_local,
                                        Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX])
                WakeMarker_A = BDpoint[BDWake_Row_IDX + 1, :]
                WakeMarker_B = TEPoint[BDWake_Row_IDX + 1, :]
                vindTR += Vortex_Scully(WakeMarker_A, WakeMarker_B, Marker_local,
                                        Gamma_BD[BDWake_Row_IDX], rcBD[BDWake_Row_IDX])

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


        # AerodynamicSetting
        self.BlendPosition=np.array([2.85, 4.65])*0.0254

        # Calculation Setup
        self.alt = 0 #m
        self.vFree = [0, 0, 5] #FreeVelocity
        self.nAzmuth = 36 #slice of rotation
        self.dAngle = 0
        self.RPM = 3000 #rev / min
        self.dt=0
        self.Blade = 2
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
        _ = Vortex_Scully(A, B, ColocationPoint, vortexStrength, rc)
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
        self.PannelGeom_Chorld = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom_Beta = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom_Thickness = np.zeros([self.inputGeom.shape[0] - 1, 1])
        self.PannelGeom =np.zeros([self.inputGeom.shape[0] - 1, 5])
        self.collocation_point_local = np.zeros([self.inputGeom.shape[0] - 1, 2])

        self.Calc_PanelGeom()

        # initial BoundGammaVector
        self.Gamma_bound = np.ones([self.inputGeom.shape[0]-1, 1])
        #self.now_Gamma_bound = np.ones([self.inputGeom.shape[0]-1, 1])
        self.rc_panel=(self.PannelGeom_Chorld*0.1).copy()
        self.rc_Geom=(self.BladeChord*0.1).copy()

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
        self.Gamma_Wake_Shed = np.zeros([1,self.inputGeom.shape[0]-1])  # Shed Wake Strength
        self.WakeMarker_Vel=np.zeros_like(self.WakeMarker_Position)
            # Shed Wake Vortex

            # Disk Induced Velocity
        self.Vel_colocation_Bound_Induced_XYZ=np.zeros([self.inputGeom.shape[0]-1,3])
        self.Vel_colocation_Wake_Induced_XYZ = np.zeros([self.inputGeom.shape[0] - 1, 3])
        self.Vel_colocation_Total_XYZ = np.zeros([self.inputGeom.shape[0] - 1, 3])


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
            Section1 = self.inputGeom[idx,:]
            Section2 = self.inputGeom[idx + 1,:]
            self.PannelGeom[idx,:]=((Section1+Section2)/2).copy()
            #print(idx)
            #print(PannelGeom_tmp)
            #print()

            #PannelGeom : 각 패널의 중앙위치
            self.PannelGeom_r[idx,:]=np.array([self.PannelGeom[idx,0], self.PannelGeom[idx,1]])
            self.PannelGeom_Chorld[idx]= self.PannelGeom[idx,2]
            self.PannelGeom_Beta[idx] = self.PannelGeom[idx,3]
            self.PannelGeom_Thickness[idx] = self.PannelGeom[idx,4]
            # Twist Center is 0.25c
            # Colocation Point is 0.75c
            # LE----BD----/-----Col------TE
            # 0.25c -----0 ------0.75c----Te

            #Calculate Real Position
            #Geometry Bound Vortex Point, Leading Edge Point, TailEdge point
            #Coordinate : [Radial, Normal, Tangential]
            #기준점 : 0.25C
        r =  self.inputGeom[:,1] #local section radial position
        n = (self.inputGeom[:, 2]*0.25)*sind(self.inputGeom[:, 3])
        t = (self.inputGeom[:, 2]*0.25)*cosd(self.inputGeom[:, 3])
        self.GeomPointVector_BD_Global_ini = np.c_[r, r*0, r*0].copy() #[r, 0, 0]
        self.GeomPointVector_LE_Global_ini = np.c_[r, t, n].copy()
        self.GeomPointVector_TE_Global_ini = np.c_[r, -3 * t, -4 * n].copy()


        #Colocation
        r =  self.PannelGeom[:,1] #local section radial position
        n = (self.PannelGeom[:, 2]*0.25)*sind(self.PannelGeom[:, 3])
        t = (self.PannelGeom[:, 2]*0.25)*cosd(self.PannelGeom[:, 3])
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
                vind_Bound[gamma_idx,:]=Vortex_Scully(Prop_position_Bound[gamma_idx,:],Prop_position_Bound[gamma_idx+1,:],ColoP,gamma, self.rc_panel[gamma_idx])
                vind_trLeft[gamma_idx,:]=Vortex_Scully(Prop_position_TE[gamma_idx,:],Prop_position_Bound[gamma_idx,:],ColoP,gamma, self.rc_panel[gamma_idx])
                vind_trRight[gamma_idx,:]= Vortex_Scully(Prop_position_Bound[gamma_idx+1,:],Prop_position_TE[gamma_idx+1,:],ColoP,gamma, self.rc_panel[gamma_idx])
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
        self.now_Panl_position_LE =(Rz(Angle_deg)@self.PanlPointVector_BD_Global_ini.T).T.copy()  # x y z coord
        self.now_Panl_position_TE =(Rz(Angle_deg)@self.PanlPointVector_BD_Global_ini.T).T.copy()  # x y z coord

        # Colocation Geometry
        self.now_Colo_position = (Rz(Angle_deg)@self.collocation_point_Global_ini.T).T.copy()   # x y z coord



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

        # 외부 함수 호출
        self.Vel_colocation_Wake_Induced_XYZ = calculate_wake_induced_velocity(WakeMarker, Gamma_TR,Gamma_Shed, coloP, rc)

    def Calc_Wake_Velocity(self):
        WakeMarker = self.WakeMarker_Position.copy()
        Gamma_TR = self.Gamma_Wake_TR.copy()
        Gamma_Shed = self.Gamma_Wake_Shed.copy()
        rc = self.rc_Geom.copy()
        Gamma_BD=self.Gamma_bound
        BDpoint=self.now_Prop_position_Bound
        TEPoint=self.now_Prop_position_TE
        rcBD=self.rc_panel.copy()


        # 외부 함수 호출
        self.WakeMarker_Vel = calculate_wake_velocity(WakeMarker, Gamma_TR,Gamma_Shed,Gamma_BD,BDpoint,TEPoint, rc,rcBD)


    def WakeUpdate(self):
        #Wake의 Timestep Update
        #VLM이 다 게산되고, Updated된 Geom에 맞추어 확정된 Gamma를 이용하여 각 Point에 유도되는 속도계산
        #오일러 1차 양해법으로 앞으로 전진
        # 유도속도, Wake 유도속도에 의해 각 Wake Point의 속도 계산
        # V= FreeVel + Colocation Point Induced Vel + InducedVel의 Rotation Speed (Propeller Vortex), 각 Wake에의해 유도된속도
        #

        boundGamma = np.transpose(self.Gamma_bound.copy())
        oldMarker = self.WakeMarker_Position.copy()
        freeVel=np.ones_like(oldMarker)
        self.dt=60*(self.dAngle/(self.RPM*360.0))
        for ind in range(boundGamma.shape[1]+1):
            freeVel[:,ind*3]=(freeVel[:,ind*3]*self.vFree[0]).copy()
            freeVel[:,ind*3+1]=(freeVel[:,ind*3+1]*self.vFree[1]).copy()
            freeVel[:,ind*3+2]=(freeVel[:,ind*3+2]*self.vFree[2]).copy()

        oldMarker=oldMarker+(freeVel+self.WakeMarker_Vel)*self.dt

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

    def VLM(self):
        # Bound Gamma 추정
        # Induced Velocity 계산
        self.Vel_colocation_Total_XYZ=self.Vel_colocation_Wake_Induced_XYZ+self.Vel_colocation_Wake_Induced_XYZ

        # 추력 계산
        # Bound Gamma 계산
        # Old값과 New값이 수렴할때까지 반복 계산
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


    def calcFlowVector(self, Vel_Free_vec, Vel_induced_vec, rR, angle, ):


        pass


    def Prop_Tilt_set(self, Tilt_angle, Tilt_Phi, Azimuth):
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

    def AzimuthSet(self, NowAngle):
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


    def RadiusVectorSet(self, rR,rpm):
        r = R * rR  # 실제 블레이드의 반경 위치

        # 해당 반경에서의 회전에 의한 접선 속도벡터 계산
        Vt_rot = (2 * np.pi * (rpm/60.0) * r) * BLP_G_unit_chord_neg

        # 회전, 자유 유동, 유도 속도의 각 성분 계산
        L_Vt_Axis = 0 * BLP_G_unit_axis  # 축 방향은 0
        L_Vt_tan = -Vt_rot  # 코드 방향(항력 관련)
        L_Vt_Span = 0 * BLP_G_unit_span  # span 방향은 0

        # 자유 유동(차량 속도)의 각 성분 (지면 기준)
        L_vFree_Axis = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_axis) * BLP_G_unit_axis
        L_vFree_Chord = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_chord_neg) * BLP_G_unit_chord_neg
        L_vFree_Span = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_span) * BLP_G_unit_span

        # 유도 속도의 각 성분 (지면 기준)
        L_Vinduced_Axis = np.dot(G_v_ind_col, BLP_G_unit_axis) * BLP_G_unit_axis
        L_Vinduced_Chord = np.dot(G_v_ind_col, BLP_G_unit_chord_neg) * BLP_G_unit_chord_neg
        L_Vinduced_Span = np.dot(G_v_ind_col, BLP_G_unit_span) * BLP_G_unit_span

        # 각 성분의 합산: 블레이드 요소에서의 전체 유동 속도 성분
        L_vAxis = L_Vt_Axis + L_vFree_Axis + L_Vinduced_Axis
        L_vChord = L_Vt_tan + L_vFree_Chord + L_Vinduced_Chord
        L_vSpan = L_Vt_Span + L_vFree_Span + L_Vinduced_Span

        W_flow = np.linalg.norm(L_vAxis + L_vChord + L_vSpan)
        Phi_flow = np.atan2(-np.dot(L_vAxis, BLP_G_unit_axis), -np.dot(L_vChord, BLP_G_unit_chord_neg))
        vA = np.dot(L_vChord, BLP_G_unit_chord_neg)
        vT = np.dot(L_vAxis, BLP_G_unit_axis)

        # print(Phi_flow, W_flow)


