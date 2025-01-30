
import numpy as np
import csv

from numpy.ma.core import zeros_like

from VortexModel import Vortex_Scully
from numba import njit
def sind(x):
    return np.sin(np.deg2rad(x))
def cosd(x):
    return np.cos(np.deg2rad(x))

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
        self.vFree = [0, 0, 0] #FreeVelocity
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
    def  Calc_Setting(self):
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
        self.readGeometry()
        return np.array(numeric_data)

    def readGeometry(self):
        self.R = self.inputGeom[self.inputGeom.shape[0]-1,1]
        self.bmean = np.median(self.inputGeom[:, 2])
        self.Calc_Setting()


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