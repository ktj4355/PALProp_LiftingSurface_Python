import random
import time

import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np

# Fixed Wing Based

#              ▲ [Ascend, Disk Tangential]
#    \ - /
#  /  \ /  \
# 〓〓〓 ○ 〓〓   ▶ [Forward, Disk Normal]
#  \  / \  /
#    / - \      ∠ (Tilt Angle = 0 deg is Forward)


# Drone Based

#              ▲ [Ascend, Disk Tangential]
#    \ - /
#  /  \ /  \
# 〓〓〓 ○ 〓〓   ▶ [Z]
#  \  / \  /
#    / - \      ∠ (Tilt Angle = 90 deg is Forward)
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


def sind(angle):
    return np.sin(np.deg2rad(angle))
def cosd(angle):
    return np.cos(np.deg2rad(angle))
def tand(angle):
    return np.tan(np.deg2rad(angle))
start= time.time()

# Prop Installation Condition
TiltAngle_Fixedwing=0  # 90deg is perpendicular to Forward Direction
TiltAngle_drone=90-TiltAngle_Fixedwing  # 90deg is perpendicular to Forward Direction
#PropTiltAngle=TiltAngle_Fixedwing
Prop_TiltAngle=0# Drone
Prop_TiltPhi=0# Forward Direction

# Vehicle Condition
vehicle_speed_x = 0 # Forward (phi = 0)
vehicle_speed_y = 0 # Side    (phi = 90)
vehicle_speed_z = -50 # Ascend  (상승속도)
G_Vehicle_Speed_vec=np.array([vehicle_speed_x, vehicle_speed_y, vehicle_speed_z])
G_Inflow_Vehicle_vec=-G_Vehicle_Speed_vec
#Freestream_Foward  = -Vehicle_Speed_Forward
#Freestream_Foward  = -Vehicle_Speed_Forward



zAxisAngle=Prop_TiltPhi
yAxisAngle=Prop_TiltAngle



# Prop Rotation Matirx
pitch = np.deg2rad(yAxisAngle)
yaw = np.deg2rad(zAxisAngle)
Rp = np.array([[np.cos(pitch), 0, np.sin(pitch)],
      [0,           1,       0],
      [-np.sin(pitch), 0, np.cos(pitch)]])
Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0],
      [np.sin(yaw), np.cos(yaw), 0],
      [0, 0, 1]])
Rot_mat = Ry@Rp


# Ref Unit Vector : X Y Z
groundUnit_X=np.array([1,0,0])
groundUnit_Y=np.array([0,1,0])
groundUnit_Z=np.array([0,0,1])

# vehicle Tilt Condition

# Disk Fram Normal Vector
Unit_Disk_tanX=Rot_mat@groundUnit_X # Azimuth Angle = 0
Unit_Disk_tanY=Rot_mat@groundUnit_Y # Azimuth Angle = 90
Unit_Disk_Axis=Rot_mat@groundUnit_Z

# Disk Frame inflow Vector
G_Inflow_Disk_Axis=np.dot(G_Inflow_Vehicle_vec, Unit_Disk_Axis)
G_Inflow_Disk_tanX=np.dot(G_Inflow_Vehicle_vec, Unit_Disk_tanX)
G_Inflow_Disk_tanY=np.dot(G_Inflow_Vehicle_vec, Unit_Disk_tanY)

# Disk Inflow Element Vector on Disk Frame [axis, tanx, tany]
D_Inflow_Disk_vec=np.array([G_Inflow_Disk_tanX,G_Inflow_Disk_tanY,G_Inflow_Disk_Axis])

# Disk Inflow Element Vector on Ground Frame
G_Inflow_Disk_Axis_vec=G_Inflow_Disk_Axis*Unit_Disk_Axis
G_Inflow_Disk_tanX_vec=G_Inflow_Disk_tanX*Unit_Disk_tanX
G_Inflow_Disk_tanY_vec=G_Inflow_Disk_tanY*Unit_Disk_tanY


fig = plt.figure(1).add_subplot(projection='3d')

R=1.25
fig.axis([-R,R, -R,R, -R,R])
Now_Azimuth_Angle=0
azm=np.linspace(0,360,36)
refposition = Rot_mat @ np.array([R, 0, 0])
refposition = np.vstack((np.array([0, 0, 0]), refposition))
print("refposition : ", refposition)
fig.plot(refposition[:, 0], refposition[:, 1], refposition[:, 2])


for azmIdx in range(azm.shape[0]):
    #print()
    #print()
    Now_Azimuth_Angle=azm[azmIdx]
    #print ("Angle : ",Now_Azimuth_Angle)



    RAzimuth = np.array(    [[cosd(Now_Azimuth_Angle), -sind(Now_Azimuth_Angle), 0],
                             [sind(Now_Azimuth_Angle), cosd(Now_Azimuth_Angle), 0],
                             [0, 0, 1]])

    D_ReferenceVector=[R,0,0]
    BLP_Position=RAzimuth@D_ReferenceVector
    BLP_unit_chord_neg=RAzimuth@[0,1,0] #BLP 2D : x
    BLP_unit_span=RAzimuth@[1,0,0] #BLP 2D : out Direction
    BLP_unit_axis=RAzimuth@[0,0,1] #BLP 2D : y

    BLP_unit_x=RAzimuth@[1,0,0] #Drag direction
    BLP_unit_y=RAzimuth@[0,1,0] #out Direction
    BLP_unit_z=RAzimuth@[0,0,1] #Thrust Direction

    BLP_G_unit_chord_neg=Rot_mat@BLP_unit_chord_neg
    BLP_G_unit_span=Rot_mat@BLP_unit_span
    BLP_G_unit_axis=Rot_mat@BLP_unit_axis

    BLP_inflow_Chord_neg=np.dot(D_Inflow_Disk_vec,BLP_unit_chord_neg)
    BLP_inflow_Span=np.dot(D_Inflow_Disk_vec,BLP_unit_span)
    BLP_inflow_Axis=np.dot(D_Inflow_Disk_vec,BLP_unit_axis)

    BLP_inflow_chord_neg_vec=BLP_unit_chord_neg*BLP_inflow_Chord_neg
    BLP_inflow_span_vec=BLP_unit_span*BLP_inflow_Span
    BLP_inflow_axis_vec=BLP_unit_axis*BLP_inflow_Axis

    #print("Axis : ",BLP_inflow_Axis)
    #print("Chord : ",BLP_inflow_Chord_neg)
    #print("Span : ",BLP_inflow_Span)

    # Blade Load Model
    beta=30
    Local_chold=0.1
    rpm=500
    n=rpm/60
    rR=np.linspace(0,1,10)

    G_v_ind_col=np.array([0,0,-10])


    D_Vax_induce=np.dot(G_v_ind_col,Unit_Disk_Axis)
    D_Vtx_induce=np.dot(G_v_ind_col,Unit_Disk_tanX)
    D_Vty_induce=np.dot(G_v_ind_col,Unit_Disk_tanY)
    Disk_Vinduced_Vector=np.array([D_Vtx_induce,D_Vty_induce,D_Vax_induce])

    BLP_Vinduced_chord_neg=np.dot(Disk_Vinduced_Vector,BLP_unit_chord_neg)
    BLP_Vinduced_span=np.dot(Disk_Vinduced_Vector,BLP_unit_span)
    BLP_Vinduced_axis=np.dot(Disk_Vinduced_Vector,BLP_unit_axis)



    #print("Chord_induced : ",BLP_Vinduced_chord_neg)
    #print("Span_induced : ",BLP_Vinduced_span)

    #print("---------------------")

    #print("Axis_induced : ", np.dot(G_v_ind_col,BLP_G_unit_axis))
    #print("Chord_induced : ", np.dot(G_v_ind_col,BLP_G_unit_chord_neg))
    #print("Span_induced : ",np.dot(G_v_ind_col,BLP_G_unit_span))

    import numpy.linalg as LA
    for idx in range(rR.shape[0]):

        r=R*rR[idx]
        Vt_rot = (2 * np.pi * n * r)*BLP_G_unit_chord_neg

        L_Vt_inflow_Axis=0              *BLP_G_unit_axis
        L_Vt_inflow_Chord = -Vt_rot
        L_Vt_inflow_Span =0             *BLP_G_unit_span

        L_vFree_Axis = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_axis)         *BLP_G_unit_axis
        L_vFree_Chord = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_chord_neg)   *BLP_G_unit_chord_neg
        L_vFree_Span = np.dot(G_Vehicle_Speed_vec, BLP_G_unit_span)         *BLP_G_unit_span

        L_Vinduced_Axis = np.dot(G_v_ind_col, BLP_G_unit_axis)          *BLP_G_unit_axis
        L_Vinduced_Chord =np.dot(G_v_ind_col, BLP_G_unit_chord_neg)     *BLP_G_unit_chord_neg
        L_Vinduced_Span = np.dot(G_v_ind_col, BLP_G_unit_span)          *BLP_G_unit_span

        L_vAxis =   L_Vt_inflow_Axis    +   L_vFree_Axis    +   L_Vinduced_Axis
        L_vChord=   L_Vt_inflow_Chord   +   L_vFree_Chord   +   L_Vinduced_Chord
        L_vSpan =   L_Vt_inflow_Span    +   L_vFree_Span    +   L_Vinduced_Span
        #print("inflow",L_Vt_inflow_Axis)
        #print("Free", L_vFree_Axis)
        #print("induced", L_Vinduced_Axis)


        for temp in range(1):

            thickness=random.uniform(5,10)
            RE = random.uniform(25000, 150000)
            AOA = random.uniform(-12, 12)
            data = Aeromodel(temp%3, thickness, RE, AOA)
            #print(data)
        scl=1
        L_vAxis=L_vAxis*scl
        L_vChord=L_vChord*scl
        L_vSpan=L_vSpan*scl
        position=Rot_mat@(RAzimuth@np.array([r,0,0]))

        #print(Vt_rot)
        #print("r : ",r)
        #fig.quiver(position[0], position[1], position[2], vehicle_speed_x, vehicle_speed_y, vehicle_speed_z,color='k', length=0.1, normalize=True)
        #fig.quiver(position[0], position[1], position[2], G_v_ind_col[0], G_v_ind_col[1], G_v_ind_col[2],  color='r', length=0.1, normalize=True)
        fig.quiver(position[0], position[1], position[2], L_vAxis[0], L_vAxis[1], L_vAxis[2],  color='r')
        fig.quiver(position[0], position[1], position[2], L_vChord[0], L_vChord[1], L_vChord[2], color='g')
        fig.quiver(position[0], position[1], position[2], L_vSpan[0], L_vSpan[1], L_vSpan[2],  color='b')
        #print(np.cross(L_vChord, L_vSpan)/LA.norm(np.cross(L_vChord, L_vSpan))-BLP_G_unit_axis)
    W_flow = np.linalg.norm(L_vAxis + L_vChord + L_vSpan)
    Phi_flow = np.atan2(-np.dot(L_vAxis, BLP_G_unit_axis), -np.dot(L_vChord, BLP_G_unit_chord_neg))
    #print(Phi_flow, W_flow)
    plt.figure(2)
    plt.scatter(Now_Azimuth_Angle,np.dot(L_vAxis, BLP_G_unit_axis),color='r')
    plt.scatter(Now_Azimuth_Angle,np.dot(L_vChord, BLP_G_unit_chord_neg),color='g')
    plt.scatter(Now_Azimuth_Angle,np.dot(L_vSpan, BLP_G_unit_span),color='b')
    plt.axis('equal')
    plt.figure(5)

    vA=np.dot(L_vChord, BLP_G_unit_chord_neg)
    vT=np.dot(L_vAxis, BLP_G_unit_axis)

    plt.quiver(-vA,-vT,vA,vT,angles='xy', scale_units='xy', scale=1)
    print(np.dot(L_vChord, BLP_G_unit_chord_neg), np.dot(L_vAxis, BLP_G_unit_axis))
    plt.axis('equal')
    plt.axis([vT*2,-vT*2, vA*2, -vA*2])
print(f"{time.time() - start:.4f} sec")
print()

plt.show()
