import numpy as np
FreeVel=np.array([0,0,-15])
induced =np.array( [0,0,-5])
inflowVel=FreeVel+induced
r=np.array([0.5,0,0])
rpm=1000
radialSpd=(rpm*2*3.14)/60 #rad/s
rotAxis=np.array([0,0,1])
radialSpd_vec=radialSpd*rotAxis


globalTan_vel=radialSpd*r #m/s, cross product

print(radialSpd)
print(globalTan_vel)
print()
gobalTan_vel_vec=np.cross(radialSpd_vec,r)
airTanVel_vec= (-gobalTan_vel_vec)
FlowVec_total= FreeVel+airTanVel_vec

sectionVec_e1=r
print(FlowVec_total)
print(np.arctan2(FlowVec_total[2],FlowVec_total[2]))
