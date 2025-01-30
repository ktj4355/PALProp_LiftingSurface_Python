import sys
import time

import VLMcore
from PyQt5 import uic
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
form_class = uic.loadUiType("maingui.ui")[0]
R1=VLMcore.Rotor()
import numpy as np
from matplotlib import pyplot as plt
class MainClass(QMainWindow, form_class):

    def __init__(self):
        QMainWindow.__init__(self)
        # 연결한 Ui를 준비한다.
        self.ui=uic.loadUi("maingui.ui",self)
        self.setupUi(self)
        self.setupUIFunc()
        # 하단에 요소를 연결할 함수를 입력

        # 화면을 보여준다.
        self.show()

    def setupUIFunc(self):

        self.testBtn.clicked.connect(self.Readbtn_Click)
        self.btn_fileopen.clicked.connect(self.btn_fileopen_Click)
        self.btn_VLMRUNTEST.clicked.connect(self.WakeUpdate_test)
        self.PlotWidget.canvas.draw()
        self.RotationTest.clicked.connect(self.rotationTest)
        self.btn_Reset.clicked.connect(self.reset)
    def nowBladePlot(self):
        geompoint_LE = R1.now_Prop_position_LE
        geompoint_BD = R1.now_Prop_position_Bound
        geompoint_TE = R1.now_Prop_position_TE
        geompoint_Colo = R1.now_Colo_position

        disp = self.PlotWidget.canvas.axes
        #disp.cla()
        disp.plot(geompoint_LE[:, 0], geompoint_LE[:, 1], geompoint_LE[:, 2])
        disp.plot(geompoint_BD[:, 0], geompoint_BD[:, 1], geompoint_BD[:, 2])
        disp.axes.plot(geompoint_TE[:, 0], geompoint_TE[:, 1], geompoint_TE[:, 2])
        disp.scatter(geompoint_Colo[:, 0], geompoint_Colo[:, 1], geompoint_Colo[:, 2])

        disp.set_aspect('equal', 'box')

    def rotationTest(self):
        angle=np.linspace(0,360,36)

        for deg in angle:
            R1.Calc_Blade_Rotation_Position_XYZ(deg)
            #print(deg)
            self.nowBladePlot()
        self.PlotWidget.canvas.draw()

    def Readbtn_Click(self):
        fPath=self.txt_GeomFileName.toPlainText()
        A=R1.readImportFile(fPath)

        #A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        #print(R1.bmean)
        #print(R1.collocation_point_Global_ini)
        self.wd.setRowCount(A.shape[0])
        self.wd.setColumnCount(A.shape[1])
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                pass
                self.wd.setItem(row, col, QTableWidgetItem(str(A[row, col])))

        PannelPoint=R1.PannelGeom
        self.TB_PenneGeomView.setRowCount(PannelPoint.shape[0])
        self.TB_PenneGeomView.setColumnCount(PannelPoint.shape[1])
        for row in range(PannelPoint.shape[0]):
            for col in range(PannelPoint.shape[1]):
                pass
                self.TB_PenneGeomView.setItem(row, col, QTableWidgetItem(str(PannelPoint[row, col])))

        self.nowBladePlot()
        self.PlotWidget.canvas.draw()

    def btn_fileopen_Click(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './',"CSV Files (*.csv *.CSV)")
        print(fname[0])
        self.txt_GeomFileName.setText(fname[0])

    def WakeUpdate_test(self):
        Rotcnt = 0
        totalAngle = 0
        start= time.time()
        while(1):
            dAngle=10

            nowAngle=R1.now_Angle
            totalAngle=totalAngle+dAngle
            newAngle=nowAngle+dAngle

            if newAngle>=360:
                newAngle=newAngle-360
                Rotcnt=Rotcnt+1




            if totalAngle/360 > 1:
                break;
            #R1.Calc_Bound_Induced_Velocity_XYZ()
            R1.Calc_Wake_Induced_Velocity_XYZ()
            R1.VLM()
            R1.Calc_Wake_Velocity()
            R1.Calc_Blade_Rotation_Position_XYZ(newAngle)
            R1.WakeUpdate()


            print(totalAngle/360)
            print(np.mean(R1.Vel_colocation_Total_XYZ[:,2]))
            print(f"{time.time() - start:.4f} sec")
            print()
        #print(totalAngle)
        print(f"{time.time() - start:.4f} sec")
        geompoint_LE = R1.now_Prop_position_LE
        geompoint_BD = R1.now_Prop_position_Bound
        geompoint_TE = R1.now_Prop_position_TE
        geompoint_Colo = R1.now_Colo_position

        nowWakePoint = R1.WakeMarker_Position.copy()

        disp = self.PlotWidget.canvas.axes

        disp.cla()
        disp.plot(geompoint_LE[:, 0], geompoint_LE[:, 1], geompoint_LE[:, 2])
        disp.plot(geompoint_BD[:, 0], geompoint_BD[:, 1], geompoint_BD[:, 2])
        disp.axes.plot(geompoint_TE[:, 0], geompoint_TE[:, 1], geompoint_TE[:, 2])
        disp.scatter(geompoint_Colo[:, 0], geompoint_Colo[:, 1], geompoint_Colo[:, 2])
        for ind in range(int(nowWakePoint.shape[1] / 3)):
            tePoint_XYZ = nowWakePoint[:, ind * 3:ind * 3 + 3]
            # print(nowWakePoint)
            # print(tePoint_XYZ)
            disp.plot(tePoint_XYZ[:, 0], tePoint_XYZ[:, 1], tePoint_XYZ[:, 2], 'k')
        disp.set_aspect('equal', 'box')

        self.PlotWidget.canvas.draw()
        #print(R1.Vel_colocation_Wake_Induced_XYZ)




    def VortexLattice_GO(self):
        R1.Calc_Blade_Rotation_Position_XYZ(0)
        R1.Calc_Iij_BoundMatrix()
        R1.VLMset()
        residual=10000
        while (residual<0.001):

            R1.Calc_Bound_Induced_Velocity_XYZ()
            R1.Calc_Wake_Induced_Velocity_XYZ()
            R1.Calc_PropForce()
            residual=R1.VLM_GammaUpdate()

        pass
        WakeUpdate(self)

    def reset(self):

        R1. __init__()
        self.PlotWidget.canvas.axes.cla()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainClass()
    app.exec_()