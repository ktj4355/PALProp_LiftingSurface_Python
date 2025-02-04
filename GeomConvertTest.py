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
def sind(angle):
    return np.sin(np.deg2rad(angle))
def cosd(angle):
    return np.cos(np.deg2rad(angle))
def tand(angle):
    return np.tan(np.deg2rad(angle))

# Prop Installation Condition
TiltAngle_Fixedwing=0  # 90deg is perpendicular to Forward Direction
TiltAngle_drone=90-TiltAngle_Fixedwing  # 90deg is perpendicular to Forward Direction
#PropTiltAngle=TiltAngle_Fixedwing
Prop_TiltAngle=90 # Drone
Prop_TiltPhi=0 # Forward Direction

# Vehicle Condition
Vehecle_Speed_x = 12 # Forward (phi = 0)
Vehecle_Speed_y = 0 # Side    (phi = 90)
Vehecle_Speed_z = 0 # Ascend  (상승속도)
G_Vehecle_Speed_vec=np.array([Vehecle_Speed_x,Vehecle_Speed_y,Vehecle_Speed_z])
G_Inflow_Vehecle_vec=-G_Vehecle_Speed_vec
#Freestream_Foward  = -Vehecle_Speed_Forward
#Freestream_Foward  = -Vehecle_Speed_Forward



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
G_Inflow_Disk_Axis=np.dot(G_Inflow_Vehecle_vec,Unit_Disk_Axis)
G_Inflow_Disk_tanX=np.dot(G_Inflow_Vehecle_vec,Unit_Disk_tanX)
G_Inflow_Disk_tanY=np.dot(G_Inflow_Vehecle_vec,Unit_Disk_tanY)

# Disk Inflow Element Vector on Disk Frame [axis, tanx, tany]
D_Inflow_Disk_vec=np.array([G_Inflow_Disk_tanX,G_Inflow_Disk_tanY,G_Inflow_Disk_Axis])

# Disk Inflow Element Vector on Ground Frame
G_Inflow_Disk_Axis_vec=G_Inflow_Disk_Axis*Unit_Disk_Axis
G_Inflow_Disk_tanX_vec=G_Inflow_Disk_tanX*Unit_Disk_tanX
G_Inflow_Disk_tanY_vec=G_Inflow_Disk_tanY*Unit_Disk_tanY

R=1.25
Now_Azimuth_Angle=90



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


BLP_inflow_Chord_neg=np.dot(D_Inflow_Disk_vec,BLP_unit_chord_neg)
BLP_inflow_Span=np.dot(D_Inflow_Disk_vec,BLP_unit_span)
BLP_inflow_Axis=np.dot(D_Inflow_Disk_vec,BLP_unit_axis)

BLP_inflow_chord_neg_vec=BLP_unit_chord_neg*BLP_inflow_Chord_neg
BLP_inflow_span_vec=BLP_unit_span*BLP_inflow_Span
BLP_inflow_axis_vec=BLP_unit_axis*BLP_inflow_Axis

print(BLP_inflow_Axis)
print(BLP_inflow_Span)
print(BLP_inflow_Chord_neg)

