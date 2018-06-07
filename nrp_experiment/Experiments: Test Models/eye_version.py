# Imported Python Transfer Function
from std_msgs.msg import Float64
@nrp.MapVariable("coordinates",  scope = nrp.GLOBAL )
@nrp.Neuron2Robot(Topic('/robot/eye_version/pos', Float64))
def eye_version(t,coordinates):
    tf = hbp_nrp_cle.tf_framework.tf_lib
    x,y = coordinates.value
    def deg2rad(deg):
        """
        Degrees to radians conversion function.
        :param deg: value in degrees
        :return: value of deg in radians
        """
        return (float(deg) / 360.) * (2. * np.pi)
    x_a,y_a = tf.cam.pixel2angle(x, y)
    return deg2rad(x_a)
##
