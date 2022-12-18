import math as m



def planet_armspeed(ratio,wfirst,wlast):
    """
    Calculates the arm speed output for a planetary gear system
    Inputs:
    ratio = gear train ratio should be ratio of intput to output speed
    wfirst = speed of the sun gear
    wlast = angular speed of the ring gear
    RETURNS: 
    warm: angular velocity of of the output arm
    """
    warm = (ratio*(wfirst)-wlast)/(ratio-1) 
    return warm

def planet_ringspeed(ratio,wfirst,warm):
    """
    Calculates the ring speed output for a planetary gear system
    Inputs:
    ratio = gear train ratio should be ratio of intput to output speed
    wfirst = speed of the sun gear
    warm = angular speed of the planet gear
    RETURNS: 
    wring: angular velocity of of the ring
    """
    wring = ratio*(wfirst-warm)+warm
    return wring

input_speed = 1800
Stage_1_output = planet_armspeed(1/10,input_speed,0)
print(Stage_1_output)
Stage_2_output = planet_ringspeed(1/10, Stage_1_output, 0)

print(Stage_2_output*0.6)

rpm = 1/(m.pi*d_drum*