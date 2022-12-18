%%%%% Functions 
%Functions created for different examples parts/key concepts

%Eqation generation functions
function eqn = moment_eqn(moments, rad_vectors, force_vectors)
% moment_eqn creates the moment equation at a point
%
%INPUTs:
%   moments: 2D array of  x y z components of moments acting on the point
%   rad_vectors:  2D array containing x y z compontents of the distance 
%                 vectors between point of rotation and line of force
%   force_vectors: 2D array containing x y z components of all forces that
%                  can cause moments
%RETURNS: 
%   eqns = 1D vector containg the equations for the moment in x y z
%         directions

%Ads components from the moments array into respective eqns
    eqn = [];
    for i = 1:size(moments,1)
        if numel(eqn) == 0
            eqn = moments(i,:).';
        else
            eqn = eqn + moments(i,:).';
        end 
    end
 %Does RxF for each item in the radius and force arrays and adds to the 
 %equation
    for i = 1:size(rad_vectors,1)
        
        temp_vector1 = rad_vectors(i,:);
        temp_vector2 = force_vectors(i,:);

        if numel(eqn) == 0
            eqn = (cross(temp_vector1,temp_vector2'))';
        else
            eqn = eqn + (cross(temp_vector1,temp_vector2'))';
        end
    end
end

function feqn = force_eqn(force_vectors)
% force_eqn generates a 1D vector containing x y z sum of forces equations
%
%INPUTS:
%   force_vectors:2D array containing x y z components of all forces
%
%RETURNS:
%    feqn: a 1D vector with x y z components of sum of forces

%Takes each row of the vector and sums it and adds to the feqn vector 
    feqn = [];
    for i = 1:size(force_vectors,1)
        if numel(feqn) == 0
            feqn = force_vectors(i,:);
        else
           feqn = feqn + force_vectors(i,:);
        end 
    end
end

%Example 3 Functions

function preload = bolt_preload(k,d,T)
    preload = T/(k*d);
end

function tcoef = torq_coef(d,p,alpha,ft,fc)
%just the torque equation from example 3
%d = major diameter of bolt(in)
%pitch = pitch of the bolt(in)
%alpha = thread angle(deg)
%ft = coefficient of frinction on threads
%fc = Coefficient of friction on collar
    
    dr = d - 1.299038*p
    dm = (d+dr)/2
    lambda = atand(p/(pi*dm))
    
    tcoef = (dm/(2*d))*((tand(lambda)+ft*secd(alpha))/(1-ft*tand(lambda)*secd(alpha)))+0.625*fc;
end

%example 4 Functions
function bstiff = bolt_stiffness(lt,ld,At,Ad,Et,Ed)
%Bolt stiffness calculations for stiffness coefficient
%lt = length of theded portion
%l
    kbt = (pi*(dt/2)^2*Et)/lt;   %Threaded stiffness
    kbd = (pi*(dd/2)^2*Ed)/ld;   %Stiffness of unthreaded region

    bstiff = ((1/kbt)+(1/kbd))^-1;
end

function mstiff = member_stiffness(Em,d,A,Bd,l)
%Member stiffness function
%this function is going to take more work
    mstiff = Em*d*(A*e^(Bd/l));
end

function [cb, cm] = boltmem_coefs(kb,km)
%Calculates portion of load carried by bolt and member
    cb = kb/(kb+km);     %bolt load
    cm = 1-cb;           %member load
end

function [bl, ml] = boltmem_loads(cb, cm, P, Fi)
%Calcualtes the forces in the member and bolt
        bl = Fi + cb*P;    %bolt Load
        ml = -Fi + cm*P;   %memeber load
        assert(bl >= P,"Seperation occured in bolt see boltmem_loads function")
end

%Example 5/6 Functions
function ax_sf = bolt_axial_sf(Sp,d,F)
%Calculates the axial stress safty factor of the bolt
       axial_stress = F/((pi*d^2)/4);
       ax_sf = Sp/axial_stress;
end

function to_sf = twistoff_sf(T,dr,Sy)
%Calculates the safty factor for torsional twist off 
    torsional_shear = T/((pi*dr^3)/16); 
    to_sf = 0.577*Sy/torsional_shear;
end

function bs_sf = boltshear_sf(F,d,Sy)
%Calculates bolt shearing SF
%Note d is either root diam or major diam depending on where shear plane
    shear_stress = F/((pi*d^2)/4);
    bs_sf = 0.577*Sy/shear_stress;
end

function ts_sf = nut_threadshear_sf(F,d,t,Sy)
%Calculates thread shearing sf from nuts
    shear_stress = F/(pi*d*0.88*t);
    ts_sf = (0.577*Sy)/shear_stress;
end

function [LF, nf] = bolt_fatigue(cb,P,Se,F,dt,Sut,Sp)
%Calculates the load factor LF and the modified goodman safty factors
%Example 7 Functions
    At = (pi*dt^2)/4;
    axial_stress = cb*P/(2*At);
    preload_stress = F/At;

    nf = (Se*(Sut-preload_stress))/(axial_stress*(Sut+Se));
    LF = (Sp-preload_stress)/(2*axial_stress);
end

function bsf = sf_bearing(F,t,d,Sy)
%Calulates the bearing stress safety factor
    bearing_stress = F/(d*t);
    bsf = Sy/bearing_stress;

end

function esf = sf_eshear(F,a,t,Sy)
%Calculates the Edge safety factor
    eshear_stress = F/(a*t);         %edge shear stress
    esf = 0.577*Sy/(eshear_stress);   %edge saftey factor
end

function mtf_sf = tf_member(F,t,w,d,Sy)
%Calculates safety factor for tensile failure of the member
        tensile_stress = F/(t*(w-d));
        mtf_sf = Sy/tensile_stress;
end

%The plan is to integrate all these safety factor functions into one
%function that will return the lowest value and what what type of failure
%it comes from

%Assignment 2 Code

%Example 8

%Maybe change this to output potential gear teeth for desired ratio
function ratio = gear_train(driven_teeth,driving_teeth)
%Calculates the train value e for a gear train
%
%INPUTs:
%   driven_teeth: 1D array of driven teeth numbers
%   driving_teeth: 1D array of driving teeth numbers
%RETURNS: 
%   e: The gear train ratio for the specified teeth numbers
        ratio = prod(driving_teeth)/prod(driven_teeth);
end

%Example 9
function [validity, msg] = gear_interferance(driving_teeth,driven_teeth,varargin)
%Calculates the validity from interferance for each of the gear pairs in a
%gear train, assumes 20deg Pressure Ratio
%
%INPUTs:
%   driving_teeth = 1D array of all the driving teeth numbers
%   driven_teeth =  1D array of all the driven teeth numbers

%RETURNS: 
%   Validity: true(1) or false(0) if the train meets teeth requirements
%   msg: array of messages for which pairs have interferance
    
validity = 1; % Valid by default
msg = zeros(1,size(driving_teeth));

%adjust these to remove assumptions
%interferance values for 20 deg full depth 
    theta = 20 ; 
    k = 1;
    
%Calculates each driving gear the max teeth that it can mate with and checks if it meets constraints   
    for i = 1:driving_teeth
        small_teeth = driving_teeth(i);
        max_teeth = (small_teeth^2*sind(theta)^2-4*k^2)/(4*k-2*small_teeth*sind(theta)^2);
        if driven_teeth(i)>max_teeth
            validity = 0;
            msg(i) = "Interference with gear " + string(small_teeth);
        else
            msg(i) = "Valid Pair";
        end
    end
end

function warm = planet_armspeed(ratio,wfirst,wlast)
%Calculates the arm speed output speed for a planetary gear system
%
%INPUTs:
%   ratio = gear train ratio
%   wfirst = speed of the sun gear
%   wlast = angular speed of the ring gear
%RETURNS: 
%   warm: angular velocity of of the output arm

warm = (wlast-ration*wfirst)/(1-ratio);
end

function wout = multi_planet_speed(gear_ratios,win)
%Calculates output speed for multi stage planteary gear system
%Assumes fixed rings
%INPUTs:
%   win = speed of the first sun gear
%   gear ratios = 1D list of gear ratios of each planetary gear
%RETURNS: 
%   warm: angular velocity of of the output arm

%Calculates arm speed for initial stage
    wout = planet_armspeed(gear_ratios(1),win,0);

%Uses previous stage to calculate output speed for next stages
    for i = 2:size(gear_ratios)
        wout = planet_armspeed(ratio(1),wout,0);
    end
end

function validity = spacing_constraints(Nring,Nsun,Planets,dpsun,dpplanets,dpring)
%Calculates the spacing constraints for a planetary gear_system
%
%INPUTs:
%   Nring = ring_teeth num
%   Nsun =  Sun teeth number
%   Planets = number of planets in the system
%   dpsun = pitch diameter of the sun
%   dpplanets = pitch diam of planets
%   dpring  = pitch diam of ring
%RETURNS: 
%   Validity: true(1) or false(0) if the train meets constraint requirements

%Valid by default
validity = 1;

x = (Nsun + Nring)/Planets;

if dpsun+2*(dpplanets) ~= dpring
    validity = 0;
end

%checks if num is int
if floor(x) ~= x
    validity = 0;

end
end

%%%%
%Assignment 3 Related Code
%%%%

%Gear Force Functions

function tin = torque(speed,hp)
% Turns a desired angluar spped and horepower to torque
% speed: Desired angular speed of output in rpm
% hp: horespower of motor
% tin: the torque available in ft lbs
tin = (63025*hp)/speed;
end

function [Ft,Fr] = gear_forces(tin,N,Pd,planets)
%Calculates the radial and tangental transmitted forces assumes 20 deg
%pressure angle
% tin = torque on gear
% N = Number of teeth
% Pd = diametral pitch
% plannets = number of planets in the system, set = 1 if not planetary
% Ft = Tangental transmitted force in lbs
% Fr = Radial Transmitted force in lbs

d = n/Pd;
Ft = (tin * d/2)/planets;
Fr = Ft*tand(20);
end

function F_planet = gear_forces_planet(Ft,Fr)
% Calculates forces in the planet of a planetary gear system asumes 0.9
% effieciency
% Ft = Tangental transmitted force in lbs
% Fr = Radial Transmitted force in lbs

Ft = Ft* 0.9;

Fr = Fr*tand(20);

F_planet = 2*Ft;
end

function T_arm = gear_forces_arm(planets,ds,dp,F_planet)
% Caluclates torque transmitted by the arm with 90% losses
%planets = number of planets
%ds = sun dear pitch diameter
%dp = planet gear pitch diameter
%F_planet = Force planet exerts on arm
%T_arm = torque transmitted by arm with losses

T_arm = planets*F_planet*0.9*((ds/2)+(dp/2));
end

%%%%
% AGMA Bending
%%%%

function [kb, kt, kr, ki, kl] = bendfatigue_adj_factors(w,Y,Pd, temp, Reliability,idler, cycles, conservative)
% Calculates the adma adjustment factors for bending given inputs
% w = width of gear
% Y = geometry factor
% Pd = diametral pitch
% temp = operation temperature
% Reliability = 99.99, 99.9, 99, 90, 50 depending on desired reliability
% idler = true for idler ger false otherwise
% cycles = cycles for intended operation
% conservative = true if a conservative approach should be taken false
%   otherwise
% kb = size factor 
% kt = temperature factor 
% kr = reliability factor
% ki = idler factor
% kl = life adjustment factor
kb = (1/1.192)*((w*sqrt(Y)/pd))^-0.0535;

kt = 1;
if temp > 250
    kt = 620/(460+temp);
end

switch Reliability
    case 99.99
        kr = 0.67;
    case 99.9
        kr = 0.8;
    case 99
        kr = 1.0;
    case 90
        kr = 1.18;
    case 50
        kr = 1.43;
end

ki = 1;

if idler
    ki = 1/1.42;
end

%may need to be adjusted if we assume less than 10^7 cycles
if cycles == 1000000
    kl = 1;

elseif conservative
    kl = 1.6831*cycles^(-0.0323);
else 
    kl = 1.3558*cycles^(-0.0323);
end

end

function [kv, ko, km] = agma_stress_adj_factors(shock,power,AccMount,quality, velocity)
%Caluclates the adjustment factors for agma stress
%shock = 1,2 or 3 depending if shock is light moderate or heavy
%power = 1,2, or 3 depending if  it is electric motor or single cylinder
%Accmount = true if the gear was accurately mounted, false otherwise
%quality = 1 to 11 depending on quality standard
%velocity = pitch line velocity of gear under analysis

%ko factor from key concepts
ko_array = [1,1.25,1.75;1.25,1.50,2.0;1.5,1.75,2.25];

ko = ko_array(power,shock);
       
km = xxx %Jd said he would tell us how to get the curves into matlab a separate time


switch quality
    case 1
        kv = (600+velocity)/600
    case 2
        kv = (1200+velocity)/1200
    case {3,4,5}
        kv = (50 + sqrt(velocity)/50);
    case {6,7,8,9,10}
        kv = (78 + sqrt(velocity)/78);
    case 11
        kv = sqrt((78+sqrt(velocity))/78)
end
end

function SF = agma_bending_sf(Sfbp, w,Y,J,Ft,Pd, temp, Reliability,idler, cycles, conservative, shock,power,AccMount,quality,velocity)

%Calls agma functions and determines a bending safety factor
% shock = 1,2 or 3 depending if shock is light moderate or heavy
% power = 1,2, or 3 depending if  it is electric motor or single cylinder
% Accmount = true if the gear was accurately mounted, false otherwise
% quality = 1 to 11 depending on quality standard
% velocity = pitch line velocity of gear under analysis
% w = width of gear
% Y = lewis geometry factor
% J = agma geometry factor
% Ft = tangental transmitted force
% Pd = diametral pitch
% temp = operation temperature
% Reliability = 99.99, 99.9, 99, 90, 50 depending on desired reliability
% idler = true for idler ger false otherwise
% cycles = cycles for intended operation
% conservative = true if a conservative approach should be taken false
%   otherwise
%SF Agma bending SF

kb, kt, kr, ki, kl = bendfatigue_adj_factors(w,Y,Pd, temp, Reliability,idler, cycles, conservative);

%find agma bending strength with factors
sfb = kb*kt*kr*ki*kl*Sfbp;

kv, ko, km = agma_stress_adj_factors(shock,power,AccMount,quality, velocity);

stress = (kv*ko*km*Ft*Pd)/(w*J);

SF = sfb/stress;
end


%%%%
%AGMA Surface Failure
%%%%

function SF = agma_surface_sf(sfcp,Cp,d,ratio,w,Y,Pd, temp, Reliability, cycles, conservative,shock,power,AccMount,quality, velocity)
%Calculates surface failure safety factor
%Assumes gears are made of the same material Ch=1 and > 10^7 cycles
%Assumes 20 deg pressure angle

% shock = 1,2 or 3 depending if shock is light moderate or heavy
% power = 1,2, or 3 depending if  it is electric motor or single cylinder
% Accmount = 1 if the gear was accurately mounted, 0 otherwise
% quality = 1 to 11 depending on quality standard
% velocity = pitch line velocity of gear under analysis
% w = width of gear
% Y = lewis geometry factor
% ratio = gear ratio between the two gears
% Cp = Material and Modulus of elasticity
% d = pitch diameter of smaller gear
% Ft = tangental transmitted force
% Pd = diametral pitch
% temp = operation temperature
% Reliability = 99.99, 99.9, 99, 90, 50 depending on desired reliability
% cycles = cycles for intended operation
% conservative = true if a conservative approach should be taken false
%   otherwise
%SF = AGMA surface failure SF
Ch = 1;

I = (cosd(20)*sind(20)/2)*(ratio/(ratio+1));

kb, kt, kr, ki, kl = bendfatigue_adj_factors(w,Y,Pd, temp, Reliability, 0, cycles, conservative,);


kv, ko, km = agma_stress_adj_factors(shock,power,AccMount,quality, velocity);

if conservative:
    Cl = 2.466*cycles^(-0.056);
else
    Cl = 1.4488*cycles^(-0.023);
end

sfc = kb*kt*kr*Cl*Ch*sfcp;

stress = Cp * sqrt((kv*ko*km)*Ft/(w*d*I));

SF = (sfc/stress)^2;
end


%%%%
%Assignment 4 related code
%%%%

function [SFS,SFF] = wrope_SF(load,psultwire,d_rope,D_drum,acceeleration)
%Calculates the static stafety factor for a 6x19 monitor steel wire rope,
%Assumes no friction, and monitor steel, Neglect weight of wire ropes
%Sult can be changed for dif steel types, 
%d_rope = rope diameter
%load = load_on roap
%psultwire = number from KC wire ropes chart
%D_drum = diameter of smallest drum wire rope is being wrapped around
%Acceleration = Acceleration the rope goes through

%change these values for different assumtions
Sult_wire = 240E3;
Sult_rope = 106E3;

E_rope = 12E6;
d_wire = 0.067*d_rope;
A_metal = 0.4*d_rope^2;

F_bend = (E_rope*d_wire*A_metal)/(D_drum);

F_fatigue = (psultwire*Sult_wire*d_rope*D_drum);

F_tension = load*(1+acceleration/32.2);

F_ult = Sult_rope*((pi*d_rope^2)/4);

%Static Safety factor
SFS = (F_ult - F_bend)/(F_tension);

%Fatigue Safety Factor
SFF = (F_fatigue-F_bend)/F_tension;
end

%%%%
%Assignment 5 Code
%%%%

function [rc, rn, e,sf] = cb_radius(bo1,bi,bo2,h1,h2,ri,ro,sy,f,meshedge)
% calculates rc, rn, and e, for shape defined in example 18. 
% bo1 = the outer side length of the shape furster from the centre of curvature
% bi = the side length of the middle (between the two shapes)
% bo2 = side length of the innermost edge of the shape
% ro = distance from centre of curvature to outside of total shape
% ri = distance from centre of curvature to inside of total shape
% h1 = length of outer part of shape
% h2 = length of inner part of shape
% f = load applied (in lbs) to hook
% sy = yield strength
% meshedge = number of divisions side area will be calculated by
% sf = yielding safety factor for hook
% rc = distance from centre of curvature to centroid of total shape 
% rn = distance from centre of curvature to the neutral axis of the total shape
% e = distance between the centroid and neutral axis for the total shape

delta_s1 = h1/meshedge;
sum_bdeltas = 0;
sum_bdeltasr = 0;
sum_rbdeltas = 0;

% numerically calculates the outer shape portion of rn and rc
for i = 0:(meshedge - 1)
    b1 = bo1+((bi-bo1)/h1)*delta_s1*i;
    sum_bdeltas = sum_bdeltas + b1*delta_s1;
    r = ro - delta_s1*i-delta_s1*0.5;
    sum_bdeltasr = sum_bdeltasr+(b1*delta_s1)/r;
    sum_rbdeltas = sum_rbdeltas + r*b1*delta_s1;
end

delta_s2 = h2/meshedge;

% numerically calculates the inner shape portion of rn and rc
for i = 0:(meshedge - 1)
    b2 = bi-((bi-bo2)/h2)*delta_s2*i;
    sum_bdeltas = sum_bdeltas + b2*delta_s2;
    r = ro - h1 - delta_s2*i-delta_s2*0.5;
    sum_bdeltasr = sum_bdeltasr+(b2*delta_s2)/r;
    sum_rbdeltas = sum_rbdeltas + r*b2*delta_s2;
end

rn = sum_bdeltas/sum_bdeltasr;
rc = sum_rbdeltas/sum_bdeltas;
e = rc-rn;

% safety factor calculation
m = f*rc;
sigma_i = m*(rn-ri)/(sum_bdeltas*e*ri);
sigma_o = m*(rn-ro)/(sum_bdeltas*e*ro);
sigma_a = f/sum_bdeltas;
sigma = max(abs(sigma_o+sigma_a,sigma_i+sigma_a));
sf = sy/sigma;

end




