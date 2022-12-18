import numpy as np
import math as m

def norm(A):
    """
    Normalyzes a vector into a unit vector
    A = vector to be normalyzed
    """
    mag = m.sqrt(A[0]**2+A[1]**2+A[2]**2)
    unit = A/mag
    return unit

def bolt_preload(k,d,T):
    preload = T/(k*d)
    return preload

def torq_coef(d,p,ft,fc,alpha=30):
    """The torque equation from example 3
    d = major diameter of bolt(in)
    pitch = pitch of the bolt(in)
    alpha = thread angle(deg)
    ft = coefficient of frinction on threads
    fc = Coefficient of friction on collar"""
    dr = d - 1.226869*p
    dm = (d+dr)/2
    lamb = atand(p/(m.pi*dm))
    tcoef = (dm/(2*d))*((tand(lamb)+ft*secd(alpha))/(1-ft*tand(lamb)*secd(alpha)))+0.625*fc
    return tcoef

def bolt_stiffness(lt,ld,At,Ad,E):
    """Bolt stiffness calculations for stiffness coefficient
    lt = length of theded portion
    ld = length of unthreaded portion
    """
    kbt = (At*E)/lt    #Threaded stiffness
    kbd = (Ad*E)/ld    #Stiffness of unthreaded region

    bstiff = ((1/kbt)+(1/kbd))**-1
    return bstiff

#Not Tested
def member_stiffness(Em,d,A,B,l):
    #Member stiffness function for similar materials
    #this function is going to take more work
    mstiff = Em*d*(A*e(B*d/l)) 
    return

def boltmem_coefs(kb,km):
    """Calculates portion of load carried by bolt and member
    kb = bolt stiffness coefficient
    km = member stiffness coefficient
    cb = percentage of load carried by bolt
    cm = percentage of load carried by member
    """
    cb = kb/(kb+km)     #bolt load
    cm = 1-cb           #member load
    return cb,cm

def boltmem_loads(cb, P, Fi):
    """
    Calcualtes the forces in the member and bolt
    cb = percentage of load carried by bolt
    cm = percentage of load carried by member
    P = applied load
    Fi = preload
    Reurns: 
    bl = load carried by bolt
    ml = load carried by member
    """ 
    cm = 1-cb
    bl = Fi + cb*P    #bolt Load
    ml = -Fi + cm*P  #memeber load
    if ml >= 0:
        ml = 0
        bl = P
    return bl, ml

#Example 5/6 Functions
def bolt_axial_sf(Sp,At,F):
    """
    Calculates the Axial stress safty factor of the bolt
    Sp = Proof strength
    At = tensile stress area
    F = Axial force 
    """
    axial_stress = F/At
    ax_sf = Sp/axial_stress
    return ax_sf

#twistoff not tested#
def twistoff_sf(T,dr,Sy):
    """
    Calculates the safty factor for torsional twist off
    T = Applied Torque
    dr = roor diameter
    Sy = yield strength
    Returns: to_sf = torsional twistoff safety factor
    """ 
    torsional_shear = T/((m.pi*dr**3)/16)
    to_sf = 0.577*Sy/torsional_shear
    return to_sf

#boltshear not tested#
def boltshear_sf(F,d,Sy):
    """
    Calculates bolt shearing SF
    d = either root diam or major diam depending on where shear plane
    F = bolt load
    Sy = Yeild strength of bolt
    Returns: bs_sf = bolt shearing safety factor
    """
    shear_stress = F/((m.pi*d**2)/4)
    bs_sf = 0.577*Sy/shear_stress
    return bs_sf

#Not Fully Tested#
def nut_threadshear_sf(F,d,Sy,p,N=1):
    """
    Calculates thread shearing sf from nuts
    F = applied force to bolt including preload
    p = thread pitch
    Sy = Yield strength of Bolt psi
    N = number of threads under load
    Returns: ts_sf = thread shear safety  factor
    """
    if N == 1:
        perc = 0.38
    else:
        perc = 1
    shear_stress = F*perc/(m.pi*d*0.88*p*N)
    print(shear_stress)
    ts_sf = (0.577*Sy)/shear_stress
    return ts_sf

#No examples to test with
def bolt_fatigue(cb,P,Se,Fi,dt,Sut,Sp):
    """
    Calculates the load factor LF and the modified goodman safty factors
    cb = cercent of load used by bolt
    dt = thread diameter of bolt
    P = extermbal load applie
    Se = endurance strength
    Fi = preload
    Sut = Ultimate Tensile Strength
    Sp = proof Strength
    Return
    nf = fatigue sf
    LF = Load Factor
    """

    At = (m.pi*dt**2)/4
    axial_stress = cb*P/(2*At)
    preload_stress = Fi/At

    nf = (Se*(Sut-preload_stress))/(axial_stress*(Sut+Se))
    LF = (Sp-preload_stress)/(2*axial_stress)
    return nf, LF

#not tested
def sf_member_other(F,t,d,Sy,a):
    """
    Calculates the bearing safty factor for a bolt
    F = shear force acting on member
    t = thickness of material
    d = diameter of hole
    sy = yield strength of material
    a = distance from hole to edge of material
    Returns:
    bsf = bearing safety factor
    esf = edge tear out safety factor
    mtf_sf = member tensile failure safety factor
    """
    bearing_stress = F/(d*t)
    bsf = Sy/bearing_stress

    eshear_stress = F/(a*t)          #edge shear stress
    esf = 0.577*Sy/(eshear_stress)    #edge saftey factor

    tensile_stress = F/(t*(w-d)) 
    mtf_sf = Sy/tensile_stress 

    return bsf, esf, mtf_sf

#Maybe change this to output potential gear teeth for desired ratio
def gear_train(driven_teeth,driving_teeth):
    """
    Calculates the train value e for a planetary gear train
    driven_teeth: 1D List of driven teeth numbers
    driving_teeth: 1D List of driving teeth numbers
    RETURNS: 
    ratio: The gear recuction ratio for the specified teeth numbers
    """
    ratio = -1
    for i in range(len(driven_teeth)):
        ratio = ratio*(driving_teeth[i]/driven_teeth[i])
    return ratio

#Example 9

#not tested
def gear_interferance(driving_teeth,driven_teeth):
    """
    Calculates the validity from interferance for each of the gear pairs in a gear train, assumes 20deg Pressure Ratio
    Inputs:
    driving_teeth = 1D array of all the driving teeth numbers
    driven_teeth =  1D array of all the driven teeth numbers
    RETURNS: 
    Validity: true or false if the train meets teeth requirements
    """
    validity = True  # Valid by default
    theta = 0.349066 # 20 deg pressure angle
    k = 1 #full depth gearing
    #Calculates each driving gear the max teeth that it can mate with and checks if it meets constraints   
    for i in range(len(driving_teeth)):
        small_teeth = driving_teeth[i]
        max_teeth = (small_teeth**2*m.sin(theta)**2-4*k**2)/(4*k-2*small_teeth*m.sin(theta)**2) 
        if driven_teeth[i] > max_teeth:
            validity = False
    return validity


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
    warm = (ratio*(wfirst-wlast))/(ratio-1) 
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

def spacing_constraints(Nring,Nsun,Nplanet,Planets,Pd):
    """
    Calculates the spacing constraints for a planetary gear_system
    Inputs:
    Nring = ring_teeth num
    Nsun =  Sun teeth number
    Nplanet = Planet teeth number
    Planets = number of planets in the system
    Pd = diametral pitch
    RETURNS: 
    Validity: true(1) or false(0) if the train meets constraint requirements
    """
    dpsun = Nsun/Pd
    dpring = Nring/Pd
    dpplanets = Nplanet/Pd
    #Valid by default
    validity = True

    x = (Nsun + Nring)/Planets 

    if dpsun+2*(dpplanets) == dpring:
        validity = False

    #checks if num is int
    if m.floor(x) != x:
        validity = False

    return validity

#Gear Force Functions

def torque(speed,hp):
    """
    Turns a desired angluar spped and horepower to torque
    speed: Desired angular speed of output in rpm
    hp: horespower of motor
    tin: the torque available in ft lbs
    """
    tin = (63025 *hp)/speed 
    return tin

def gear_forces(tin,N,Pd,planets):
    """
    Calculates the radial and tangental transmitted forces assumes 20 deg
    pressure angle and 10% loss
    tin = torque on gear
    N = Number of teeth om planet
    Pd = diametral pitch
    planets = number of planets in the system, set = 1 if not planetary
    Ft = Tangental transmitted force in lbs
    Fr = Radial Transmitted force in lbs
    """

    d = N/Pd 
    Ft = (tin * 2)/(planets*d) 
    Fr = Ft*m.tan(0.349066) 
    return 0.9*Fr, 0.9*Ft

def gear_forces_planet(Ft):
    """
    Calculates force on the arm from the planet
    Ft = Tangental transmitted force in lbs
    Fr = Radial Transmitted force in lbs
    """

    F_planet_arm = 2*Ft 

    return F_planet_arm

def gear_forces_arm(planets,ds,dp,F_planet):
    """
    Caluclates torque transmitted by the arm 
    planets = number of planets
    ds = sun dear pitch diameter
    dp = planet gear pitch diameter
    F_planet = Force planet exerts on arm
    T_arm = torque transmitted by arm with losses
    """
    T_arm = planets*F_planet*((ds/2)+(dp/2)) 
    return T_arm

####
# AGMA Bending tested until here
####

def bendfatigue_adj_factors(w,Y,Pd, temp, Reliability,idler, cycles, conservative):
    """
    Calculates the adma adjustment factors for bending given Inputs
    w = width of gear
    Y = geometry factor
    Pd = diametral pitch
    temp = operation temperature
    Reliability = 99.99, 99.9, 99, 90, 50 depending on desired reliability
    idler = true for idler ger false otherwise
    cycles = cycles for intended operation
    conservative = true if a conservative approach should be taken false
    otherwise
    kb = size factor 
    kt = temperature factor 
    kr = reliability factor
    ki = idler factor
    kl = life adjustment factor
    """
    kb = (1/1.192)*((w*m.sqrt(Y)/Pd))**-0.0535 
    assert kb > 0.817, "agma gb factor below 0.817"
    kt = 1 
    if temp > 250:
        kt = 620/(460+temp) 

    match Reliability:
        case 99.99:
            kr = 0.67 
        case 99.9:
            kr = 0.8 
        case 99:
            kr = 1.0 
        case 90:
            kr = 1.18 
        case 50:
            kr = 1.43 


    ki = 1 

    if idler:
        ki = 1/1.42 


    #may need to be adjusted if we assume less than 10**7 cycles
    if cycles <= 1000000:
        kl = 1 

    elif conservative:
        kl = 1.6831*cycles**(-0.0323) 
    else:
        kl = 1.3558*cycles**(-0.0323) 

    return [kb,kt,kr,ki,kl]

def agma_stress_adj_factors(shock,power,AccMount,quality, velocity,width):
    """
    Caluclates the adjustment factors for agma stress
    shock = 1,2 or 3 depending if shock is light moderate or heavy
    power = 1,2, or 3 depending if  it is electric motor or single cylinder
    Accmount = true if the gear was accurately mounted, false otherwise
    quality = 1 to 11 depending on quality standard
    velocity = pitch line velocity of gear under analysis
    """
    shock = shock-1
    power = power-1
    #ko factor from key concepts
    ko_array = [[1,1.25,1.75],[1.25,1.50,2.0],[1.5,1.75,2.25]] 

    ko = ko_array[power][shock] 


     

    if AccMount:
        km = y = 0.001*width**2 + 0.0171**width + 1.2613
        if width < 2:
            km =1.3
    else:
        km = 0.002*width**2 + 0.0075*width**3 + 1.579
        if width < 2:
            km = 1.6

    match quality:
        case 1:
            kv = (600+velocity)/600
        case 2:
            kv = (1200+velocity)/1200
        case 3|4|5:
            kv = (50 + m.sqrt(velocity)/50) 
        case 6|7|8|9|10:
            kv = (78 + m.sqrt(velocity))/78
        case 11:
            kv = m.sqrt((78+m.sqrt(velocity))/78)
    return ko,kv,km

def agma_bending_sf(Sfbp,Sfcp,d,w,Y,J,Ft,Pd, temp, Reliability,idler, cycles, conservative, shock,power,AccMount,quality,velocity,ratio,Cp = 2300):
    """
    Calls agma functions and determines a bending safety factor
    sfbp = gear stength before adjustment
    sfcp = gear surface strength before adjustment
    d = diameter of smaller gear
    w = width of gear
    Y = lewis geometry factor
    J = agma geometry factor
    Ft = tangental transmitted force
    Pd = diametral pitch
    temp = operation temperature deg C
    Reliability = 99.99, 99.9, 99, 90, 50 depending on desired reliability
    idler = true for idler gear false otherwise
    cycles = cycles for intended operation
    shock = 1,2 or 3 depending if shock is light moderate or heavy
    power = 1,2, or 3 depending if  it is electric motor or single cylinder
    Accmount = true if the gear was accurately mounted, false otherwise
    quality = 1 to 11 depending on quality standard
    velocity = pitch line velocity of gear under analysis
    Cp = Stress adjustment factor
    ratio = Ndriven/Ndriving
   
    conservative = true if a conservative approach should be taken false otherwise
    SF = Agma bending SF
    """

    kb, kt, kr, ki, kl = bendfatigue_adj_factors(w,Y,Pd, temp, Reliability,idler, cycles, conservative) 

    #find agma bending strength with factors
    sfb = kb*kt*kr*ki*kl*Sfbp 
    ko, kv, km = agma_stress_adj_factors(shock,power,AccMount,quality, velocity,w) 
    stress = (kv*ko*km*Ft*Pd)/(w*J) 
    SF_bend = sfb/stress 
    
    #Surface Failure
    I = (m.cos(0.349066)*m.sin(0.349066)/2)*(ratio/(ratio+1)) 
    Ch = 1 
    if conservative:
        Cl = 2.466*cycles**(-0.056) 
    else:
        Cl = 1.4488*cycles**(-0.023) 

    sfc = kb*kt*kr*Cl*Ch*Sfcp 
    stress_surface = Cp * m.sqrt((kv*ko*km)*Ft/(w*d*I))
    SF_surface = (sfc/stress_surface)**2 
    return SF_bend,SF_surface

####
#Assignment 4 related code
####

def wrope_SF(bends,d_rope,D_drum,Sult_wire=240E3,Sult_rope=106E3,E_rope=12e6,ametal =0.40,dwire = 0.0625):
    """
    Calculates the static stafety factor for a 6x19(default) monitor steel wire rope,
    Inputs:
    psultwire = number from KC wire ropes chart
    d_rope = rope diameter (not metal diameter)
    D_drum = diameter of smallest drum wire rope is being wrapped around
    Sult_wire = Strength of individual strand (psi)
    Sult_rope = strength of rope psi
    ametal = coefficent 0.38 or 0.4 from table 17-27 Ametal
    dwire = coefficent *drope to get dwire default = 1/16
    Returns:
    SFS = Static Saftey Facto
    SFF = Fatigue Safety Factor
    """

    A_metal = ametal*d_rope**2 
    D_wire = dwire*d_rope
    F_bend = (E_rope*D_wire*A_metal)/(D_drum) 

    psultwire = 1.5714*bends**(-0.512)
    F_fatigue = (psultwire*Sult_wire*d_rope*D_drum)/2 

    F_tension = 10000

    F_ult = Sult_rope*((m.pi*d_rope**2)/4) 

    if D_drum/d_rope < 30:
        loss = 0.6717*(D_drum/d_rope)**(-0.583)
        F_bend = F_ult*loss

    #Static Safety factor
    SFS = (F_ult - F_bend)/(F_tension) 

    #Fatigue Safety Factor
    SFF = (F_fatigue-F_bend)/F_tension 

    return SFS,SFF

####
#Assignment 5 Code
####

def cb_radius(bo1,bi,bo2,h1,h2,ri,ro,sy,f,meshedge):
    """
    calculates rc, rn, e, and Safety for shape defined in example 18. 
    bo1 = the outer side length of the shape furster from the centre of curvature
    bi = the side length of the middle (between the two shapes)
    bo2 = side length of the innermost edge of the shape
    ro = distance from centre of curvature to outside of total shape
    ri = distance from centre of curvature to inside of total shape
    h1 = length of outer part of shape
    h2 = length of inner part of shape
    f = load applied (in lbs) to hook
    sy = yield strength
    meshedge = number of divisions side area will be calculated by
    Returns
    sf = yielding safety factor for hook
    rc = distance from centre of curvature to centroid of total shape 
    rn = distance from centre of curvature to the neutral axis of the total shape
    e = distance between the centroid and neutral axis for the total shape
    """
    delta_s1 = h1/meshedge 
    sum_bdeltas = 0 
    sum_bdeltasr = 0 
    sum_rbdeltas = 0 

    # numerically calculates the outer shape portion of rn and rc
    for i in range(meshedge - 1):
        b1 = bo1+((bi-bo1)/h1)*delta_s1*i 
        sum_bdeltas = sum_bdeltas + b1*delta_s1 
        r = ro - delta_s1*i-delta_s1*0.5 
        sum_bdeltasr = sum_bdeltasr+(b1*delta_s1)/r 
        sum_rbdeltas = sum_rbdeltas + r*b1*delta_s1 


    delta_s2 = h2/meshedge 

    # numerically calculates the inner shape portion of rn and rc
    for i in range(meshedge - 1):
        b2 = bi-((bi-bo2)/h2)*delta_s2*i 
        sum_bdeltas = sum_bdeltas + b2*delta_s2 
        r = ro - h1 - delta_s2*i-delta_s2*0.5 
        sum_bdeltasr = sum_bdeltasr+(b2*delta_s2)/r 
        sum_rbdeltas = sum_rbdeltas + r*b2*delta_s2 


    rn = sum_bdeltas/sum_bdeltasr 
    rc = sum_rbdeltas/sum_bdeltas 
    e = rc-rn 

    # safety factor calculation
    x = f*rc 
    sigma_i = x*(rn-ri)/(sum_bdeltas*e*ri) 
    sigma_o = x*(rn-ro)/(sum_bdeltas*e*ro) 
    sigma_a = f/sum_bdeltas 
    sigma = max(abs(sigma_o+sigma_a,sigma_i+sigma_a)) 
    sf = sy/sigma 

    return e,rn,rc,sf

## Input parameters
F_winch = 10000 #Capacity of winch lbs
F_winch_vector = F_winch * norm(np.array([0,0,1]))
rope_diameter = 7/16
drum_diameter = 4
drum_diameter_w_rope = drum_diameter+rope_diameter
Motor_input_speed = 1600 #rpm
Required_torque = F_winch*drum_diameter_w_rope/2 #in*lbs
Velocity = 5 #ft/min
required_rpm = Velocity/(m.pi*drum_diameter/12)
torque_ratio = 0

#Planetary Gear Calculations
Stage_1_output = planet_armspeed(1/6, Motor_input_speed, 0)
Stage_2_output = planet_armspeed(1/7, Stage_1_output, 0)
Stage_3_output = planet_ringspeed(1/8, Stage_2_output, 0)

Output_rpm = Stage_3_output

Output_fpm = Output_rpm*(m.pi*(drum_diameter/12))
torque_ratio = (Motor_input_speed/Output_rpm)*0.6
input_torque = Required_torque/torque_ratio

print(torque_ratio)
print(input_torque)
print(Output_fpm)


origin = np.array([0,0,0])

A = np.array([-2.46,0,-1.673])
B = np.array([0,0,1.673])
C = np.array([2.46,0,-1.673])
D = np.array([-1.257,0,0])
E = np.array([1.257,0,0])
F = np.array([-2.46,0,1.279])
G = np.array([0,0,1.535])
H = np.array([2.46,0,1.279])
I = np.array([-2.362,1.378,2.559])
J = np.array([2.362,1.378,2.559])
K = np.array([0,(1.378+drum_diameter_w_rope/2),0]) #center of roller 
L = np.array([0,1.378,0])                   #point rope acts onrope


#Calculations
#find reactions at k
F_k = -F_winch_vector
u_lk = K-L
Mk = np.cross(F_winch_vector,u_lk)


