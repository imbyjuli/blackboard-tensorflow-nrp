# Imported Python Transfer Function
from std_msgs.msg import Float64
@nrp.MapVariable("coordinates",  scope = nrp.GLOBAL )
@nrp.Neuron2Robot(Topic('/robot/eye_tilt/pos', Float64))
def tilt_eye(t, coordinates): 
    tf = hbp_nrp_cle.tf_framework.tf_lib
    x,y = coordinates.value
    x_a,y_a = tf.cam.pixel2angle(x,y)
    def deg2rad(deg):
        """
        Degrees16 to radians conversion function.
        :param deg: value in degrees
        :return: value of deg in radians
        """
        return (float(deg) / 360.) * (2. * np.pi)
    return deg2rad(y_a)
