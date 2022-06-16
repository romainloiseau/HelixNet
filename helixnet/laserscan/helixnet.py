import math
import numpy as np

from ..laserscan.semantickitti import LaserScan, SemLaserScan

class HelixNetThetaCorrector:
  """
  Class to generate the jagged profile (Fig 3. from paper)
  Everything in radian
  """
  _DJ_ = np.array([
      -5.1157565, -3.082124, 2.6482964, 4.8981576, -0.82197922, 1.4262592, -1.578348, 0.63654262,
      3.4354012, 5.694129, 2.7199235, 4.9451747, -5.1654577, -2.9371529, -5.9700422, -3.6662464,
      -0.87955129, 1.3647046, -1.6563587, 0.6224122, 3.3590462, 5.6217284, 2.6106954, 4.8653493,
      -5.2164478, -2.9751167, -5.9536142, -3.7077124, -0.93430871, 1.3023154, -1.6982456, 0.55512023,
      -7.9373188, -4.4203467, 4.3780403, 7.8267746, -0.99920338, 2.4654822, -2.2377901, 1.3167042,
      5.8131237, 9.2812138, 4.6765423, 8.2248993, -7.7736826, -4.310514, -9.1592922, -5.6445527,
      -1.0055929, 2.3561344, -2.2749445, 1.1394948, 5.5759063, 9.0024099, 4.4896321, 7.899087,
      -7.5682616, -4.3903823, -8.9081631, -5.5881448, -1.0275047, 2.2403195, -2.2508492, 1.0360665
  ]) * math.pi / 180

  _FIBER2ANGLE_ = np.array([
      36, 37, 58, 59, 38, 39, 32, 33,
      40, 41, 34, 35, 48, 49, 42, 43,
      50, 51, 44, 45, 52, 53, 46, 47,
      60, 61, 54, 55, 62, 63, 56, 57,
      4, 5, 26, 27, 6, 7, 0, 1,
      8, 9, 2, 3, 16, 17, 10, 11,
      18, 19, 12, 13, 20, 21, 14, 15,
      28, 29, 22, 23, 30, 31, 24, 25
  ])

  _INV_ = np.argsort(_FIBER2ANGLE_)

  _THETA_CORRECTION_ = _DJ_[_INV_]

  _MAX_DIFF_ = np.max(_DJ_) - np.min(_DJ_)

class LaserScanHNet(LaserScan):

  def reset(self):
    super(LaserScanHNet, self).reset()
    """ Reset scan members. """
    self.rtz = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: r, t, z
    self.time = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: time
    self.fiber = np.zeros((0, 1), dtype=np.float32)

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    scan = self.open_file(filename)

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    rtz = scan[:, 3:6]
    remissions = scan[:, 6]  # get remission
    fiber = scan[:, 7]
    time = scan[:, 8]
    self.set_points(points, remissions, rtz, fiber, time)

  def open_file(self, filename):
      # if all goes well, open pointcloud
      scan = np.fromfile(filename, dtype=np.float32)
      scan = scan.reshape((-1, 9))

      self.r_isnot_0 = scan[:, 3] != 0
      return scan[self.r_isnot_0]

  def set_points(self, points, remissions=None, rtz=None, fiber=None, time=None):
    super(LaserScanHNet, self).set_points(points, remissions)

    if rtz is not None:
      if not isinstance(rtz, np.ndarray):
        raise TypeError("rtz should be numpy array")
      else:
        self.rtz = rtz
    else:
      self.rtz = np.zeros((points.shape[0]), dtype=np.float32)

    if fiber is not None:
      if not isinstance(fiber, np.ndarray):
        raise TypeError("fiber should be numpy array")
      else:
        self.fiber = fiber
    else:
      self.fiber = np.zeros((points.shape[0]), dtype=np.float32)

    if time is not None:
      if not isinstance(time, np.ndarray):
        raise TypeError("time should be numpy array")
      else:
        self.time = time
    else:
      self.time = np.zeros((points.shape[0]), dtype=np.float32)

class SemLaserScanHNet(LaserScanHNet, SemLaserScan):
  EXTENSIONS_LABEL = ['.bin']

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    label = np.fromfile(filename, dtype=np.uint8)
    label = label.reshape((-1))
    label = label[self.r_isnot_0]

    # set it
    self.set_label(label)