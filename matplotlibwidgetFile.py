from PyQt5.QtWidgets import*
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
class matplotlibWidget(QWidget):

    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())
        self.Toolbar=NavigationToolbar(self.canvas,self)
        vertical_layout=QVBoxLayout()
        vertical_layout.addWidget(self.Toolbar)
        vertical_layout.addWidget(self.canvas)
        vertical_layout.setContentsMargins(1, 1, 1, 1)
        self.canvas.axes=self.canvas.figure.add_subplot(projection='3d')
        self.setLayout(vertical_layout)