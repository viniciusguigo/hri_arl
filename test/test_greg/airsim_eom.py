#!/usr/bin/env python
""" airsim_eom.py:
AirSim using computer vision mode and equations of motion to control the UAV.
"""

__author__ = "Vinicius Guimaraes Goecks and Gregory Michael Gremillion"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "August 14, 2017"

# import
from PythonClient import *
import sys
import time
from math import cos, sin, tan, atan2, pi
import pygame, time           # for keyboard multi-threaded inputs
from pygame.locals import *
import datetime
import numpy as np
import cv2

class AirSimEOM(object):
    """
    Integrates the Equations of Motion (EOM) and fly UAV in AirSim using
    computer vision mode.
    """

    def __init__(self):
        # initialize airsim
        self.client = AirSimClient('127.0.0.1')
        self.control = 'joystick' # 'keyboard'

        # depth parameters and compression factors
        # depth 1/4: 36 x 64 pics
        # depth 1/8: 18 x 32 pics
        depth_width = 256
        depth_height = 144
        self.reduc_factor = 1/4 # multiply original size by this factor
        self.depth_w = int(depth_width*self.reduc_factor)
        self.depth_h = int(depth_height*self.reduc_factor)

        self.stat = True
        self.V_ref = 0
        self.original_dt = 0.02
        self.learning_agent = True # disable or enable learning agent
                                   # if disabled, can use joystick/keyboard to control

        # fix inital position at x=y=0, z=-0.15
        # pos = self.client.getPosition()
        pos = [0,0,-.15]
        orq = [1,0,0,0]
        self.pos0,self.orq0 = (pos,orq)
        self.pos_new,self.orq_new = (pos,orq)
        self.ore0 = self.client.toEulerianAngle(orq)

        print('Initial Position: ', self.pos0)

        self.rll_prv,self.pch_prv,self.yaw_prv,self.thr_prv = (0.0,0.0,0.0,0.0)
        self.drll,self.dpch,self.dyaw,self.dthr = (0.0,0.0,0.0,0.0)
        self.drll_prv,self.dpch_prv,self.dyaw_prv,self.dthr_prv = (0.0,0.0,0.0,0.0)
        self.mode_prv = 0 # 0 = independent axis control, 1 = forward velocity and coordinated turn control

        self.orq_new = [0.0,0.0,0.0,0.0]
        self.ore_new = [0.0,0.0,0.0]
        self.ore_check = [0.0,0.0,0.0]

        self.ps = 0

        self.g = 9.81 # gravity [m/s^2]
        self.m = 0.6 # mass [kg]
        self.Ix,self.Iy,self.Iz = (0.00115,0.00115,0.00598) # inertia tensor principle components [kg m^2] (based on Gremillion 2016 https://arc.aiaa.org/doi/abs/10.2514/1.J054408)
        self.Lp,self.Mq,self.Nr,self.Xu,self.Yv,self.Zw = (-0.01,-0.01,-0.2,-0.05,-0.05,-0.05) # aerodynamic drag derivatives [kg / s]

        self.K_ph,self.K_p = (0.5,-0.02)
        self.K_th,self.K_q = (self.K_ph,self.K_p)
        self.K_r,self.K_dr = (0.2,0.0)
        self.K_dps = 0.2
        self.K_z,self.K_dz,self.K_z_i = (-20.0,5.0,-0.05)
        self.K_dv = 0.5
        self.K_v = 1.0

        # initialize states
        self.ph,self.th,self.ps,self.p,self.q,self.r,self.u,self.v,self.w,self.x,self.y,self.z = (self.ore0[0],self.ore0[1],self.ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,self.pos0[0],self.pos0[1],self.pos0[2])
        # initialize state derivatives
        self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv = (self.ph,self.th,self.ps,self.p,self.q,self.r,self.u,self.v,self.w,self.x,self.y,self.z)
        # initialize state derivatives
        self.dph,self.dth,self.dps,self.dp,self.dq,self.dr,self.du,self.dv,self.dw,self.dx,self.dy,self.dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        # initialize aerodynamic states
        self.tawx,self.tawy,self.tawz,self.fwx,self.fwy,self.fwz = (0.0,0.0,0.0,0.0,0.0,0.0) # wind forces [N] torques [N m]

        self.z_i = 0.0 # z error integral

        # get initial collision status
        collision = self.client.getCollisionInfo()
        self.initial_col1 = collision[1]

        # initialize pygame for inputs
        if self.control == 'keyboard':
            pygame.display.init()
            pygame.font.init()
            screen = pygame.display.set_mode((500, 120))
            pygame.display.set_caption('CLICK HERE TO CONTROL DRONE :)')

        elif self.control == 'joystick':
            pygame.init()
            self.my_joystick = pygame.joystick.Joystick(0)
            self.my_joystick.init()

        # initialize time
        self.previous_time = time.time() + self.original_dt


    def getJoystickCommands(self,stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr):
        pygame.event.get()

        # control
        drll = self.my_joystick.get_axis(3) # 0 = right stick lat
        dpch = self.my_joystick.get_axis(4) # 0 = right stick long
        dyaw = self.my_joystick.get_axis(0) # 0 = left stick lat
        dthr = self.my_joystick.get_axis(1) # 0 = left stick long

        # options
        if self.my_joystick.get_button(6): # select button
            mode = 0

        if self.my_joystick.get_button(7): # start button
            mode = 1

        if  self.my_joystick.get_button(8): # big xbox button
            rset = True # reset position/controls
            rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
            drll,dpch,dyaw,dthr = (0.0,0.0,0.0,0.0)

        # flush rest and display
        pygame.event.clear()

        rll = drll / 4.0
        pch = dpch / 4.0
        yaw = dyaw / 1.0
        thr += dthr / 80.0


        return (stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

    def step(self, action):
        """
        Compute next states based on joystick inputs and defined equations.
        """
        # rl parameters
        info = 0

        # compute current dt
        current_time = time.time()
        dt = current_time - self.previous_time
        dt = np.clip(dt,0,self.original_dt)

        self.rset = False
        mode = self.mode_prv

        # retrieve states from buffer
        ph,th,ps,p,q,r,u,v,w,x,y,z = (self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv)

        # retrieve commands from buffer
        rll,pch,yaw,thr = (self.rll_prv,self.pch_prv,self.yaw_prv,self.thr_prv)
        drll,dpch,dyaw,dthr = (self.drll_prv,self.dpch_prv,self.dyaw_prv,self.dthr_prv)

        # get commands from keyboard
        stat,rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr = self.getJoystickCommands(self.stat,self.rset,mode,rll,pch,yaw,thr,drll,dpch,dyaw,dthr)

        # override with learning agent actions
        if self.learning_agent:
            rll = action[0] / 4
            pch = action[1] / 4

        # zero commands if mode changes
        if mode != self.mode_prv:
            rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
            tawx,tawy,tawz,fwx,fwy,fwz = (0.0,0.0,0.0,0.0,0.0,0.0)
            ph,th,p,q,r,u,v,w = (self.ore0[0],self.ore0[1],0.0,0.0,0.0,0.0,0.0,0.0)
            dph,dth,dps,dp,dq,dr,du,dv,dw,dx,dy,dz = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
            be = 0.0

        # wind angles
        al = atan2(w,u)
        be = atan2(v,u)

        # control mode case
        if mode == 0:
            # state reference
            ph_ref = 2.0 * rll
            th_ref = 2.0 * pch
            dps_ref = 10.0 * yaw

            # state error
            ph_err = ph_ref - ph
            th_err = th_ref - th
            dps_err = dps_ref - self.dps

            # force/torque control
            tax = self.K_ph * ph_err + self.K_p * p
            tay = 3.0 * self.K_th * th_err + 4.0 * self.K_q * q
            taz = self.K_dps * dps_err
        else:
            # state reference
            ph_ref = 2.0 * rll
            self.V_ref += -1.0 * pch
            dps_ref = 10.0 * be
            V = u * cos(th) + w * sin(th)
            V_err = self.V_ref - V
            th_ref = -V_err

            # state error
            ph_err = ph_ref - ph
            th_err = th_ref - th
            dps_err = dps_ref - self.dps

            # force/torque control
            tax = self.K_ph * ph_err + self.K_p * p
            tay = 3.0 * self.K_th * th_err + 4.0 * self.K_q * q
            if V <=0.1:
                taz = 0
            else:
                taz = self.K_dps * dps_err

        # heave control
        z_ref = 10.0 * thr + self.pos0[2]
        z_err = z_ref - z
        ft = self.K_z * z_err + self.K_dz * self.dz + self.K_z_i * self.z_i + self.m * self.g

        # drag terms
        tawx = self.Lp * p
        tawy = self.Mq * q
        tawz = self.Nr * r
        fwx = self.Xu * u
        fwy = self.Yv * v
        fwz = self.Zw * w

        # nonlinear dynamics
        self.dph = p + r * (cos(ph) * tan(th)) + q * (sin(ph) * tan(th))
        self.dth = q * (cos(ph)) - r * (sin(ph))
        self.dps = r * (cos(ph) / cos(th)) + q * (sin(ph) / cos(th))
        self.dp = r * q * (self.Iy - self.Iz) / self.Ix + (tax + tawx) / self.Ix
        self.dq = p * r * (self.Iz - self.Ix) / self.Iy + (tay + tawy) / self.Iy
        self.dr = p * q * (self.Ix - self.Iy) / self.Iz + (taz + tawz) / self.Iz
        self.du = r * v - q * w - self.g * (sin(th)) + fwx / self.m
        self.dv = p * w - r * u + self.g * (sin(ph) * cos(th)) + fwy / self.m
        self.dw = q * u - p * v + self.g * (cos(th) * cos(ph)) + (fwz - ft) / self.m
        self.dx = w * (sin(ph) * sin(ps) + cos(ph) * cos(ps) * sin(th)) - v * (cos(ph) * sin(ps) - cos(ps) * sin(ph) * sin(th)) + u * (cos(ps) * cos(th))
        self.dy = v * (cos(ph) * cos(ps) + sin(ph) * sin(ps) * sin(th)) - w * (cos(ps) * sin(ph) - cos(ph) * sin(ps) * sin(th)) + u * (cos(th) * sin(ps))
        self.dz = w * (cos(ph) * cos(th)) - u * (sin(th)) + v * (cos(th) * sin(ph))

        # numerically integrate states
        ph += self.dph * dt
        th += self.dth * dt
        ps += self.dps * dt
        p += self.dp * dt
        q += self.dq * dt
        r += self.dr * dt
        u += self.du * dt
        v += self.dv * dt
        w += self.dw * dt
        x += self.dx * dt
        y += self.dy * dt
        z += self.dz * dt

        ## GENERATE POSITION/ORIENTATION VISUALIZATION
        # update position/orientation
        ore_new = [ph,th,ps] # new orientation (Euler)
        orq_new = self.client.toQuaternion(ore_new) # convert to quaternion
        pos_new = [x,y,z] # new position

        # convert back to Euler (check quaternion conversion)
        ore_check = self.client.toEulerianAngle(orq_new)

        if self.rset == True:
            done = 1

        # get vehicle status
        pos = self.client.getPosition() # get current position
        orq = self.client.getOrientation() # get current orientation
        collision = self.client.getCollisionInfo() # get collision status

        if ((collision[1] != self.initial_col1) or (ore_new[0] > .8*np.pi/2) or (ore_new[1] > .8*np.pi/2)): # if collision, reset position/orientation/controls
            print('[*] Crash')
            reward = -100
            done = 1

        else: # if no collision update position/orientation
            self.client.simSetPose(pos_new,orq_new)
            reward = np.sign(pos_new[0]-pos[0]) # positive if going forward
            done = 0

        # z error integral
        self.z_i += z_err * dt

        # send states to buffer
        self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)

        # send commands to buffer
        self.rll_prv,self.pch_prv,self.yaw_prv,self.thr_prv = (rll,pch,yaw,thr)
        self.drll_prv,self.dpch_prv,self.dyaw_prv,self.dthr_prv = (drll,dpch,dyaw,dthr)
        self.mode_prv = mode

        # print debug output command/orientation
        # print("phr %f, dpsr %f, be %f, V %f, Vref %f, roll %f, phi %f, pitch %f, theta %f, yaw %f, r %f, throttle %f, z %f" %(ph_ref,dps_ref,be,V,V_ref,rll,ph,pch,th,yaw,r,thr,z))
        # print("mode %f, roll %f, pitch %f, yaw %f, throttle %f" %(mode,rll,pch,yaw,thr))

        self.previous_time = current_time

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        # check if out of bounds
        if pos[0] < -2:
            # went backward on the beginning
            reward = -100
            done = 1

        if pos[0] > 90:
            # completed track
            reward = 100
            done = 1

        return res, reward, done, info

    def reset(self):
        """
        Go to initial position and restart simulation.
        """
        random_pos0 = self.pos0
        random_pos0[1] = np.random.uniform(-14,14) # random at y
        print('[*] Respawn at ', random_pos0)

        self.client.simSetPose(random_pos0,self.orq0) # reset to origin
        pos_new,orq_new = (self.pos0,self.orq0) # update current position/orientation to origin
        rll,pch,yaw,thr = (0.0,0.0,0.0,0.0)
        ph,th,ps,p,q,r,u,v,w,x,y,z = (self.ore0[0],self.ore0[1],self.ore0[2],0.0,0.0,0.0,0.0,0.0,0.0,self.pos0[0],self.pos0[1],self.pos0[2])
        self.ph_prv,self.th_prv,self.ps_prv,self.p_prv,self.q_prv,self.r_prv,self.u_prv,self.v_prv,self.w_prv,self.x_prv,self.y_prv,self.z_prv = (ph,th,ps,p,q,r,u,v,w,x,y,z)
        self.z_i = 0.0
        self.V_ref = 0

        # get initial collision status
        collision = self.client.getCollisionInfo()
        self.initial_col1 = collision[1]

        # get next state
        img = self.grab_depth()
        res = self.preprocess(img)

        return res

    def act(self):
        """
        Temporary function to simulate a learning agent.
        In this case, just do random actions.
        """
        # take random pitch and roll actions
        drll = np.random.uniform(-1,1)
        dpch = np.random.uniform(-1,1)

        action = (drll,dpch)

        return action

    def preprocess(self, img):
        """
        Resize image. Converts and down-samples the input image.
        """
        # resize img
        res = cv2.resize(img,None,fx=self.reduc_factor, fy=self.reduc_factor, interpolation = cv2.INTER_AREA)

        # normalize image
        res = res / 255

        return res

    def grab_depth(self):
        """
        Get camera depth image and return array of pixel values.
        Returns numpy ndarray.
        """
        # get depth image
        # result = self.setImageTypeForCamera(0, AirSimImageType.Depth)
        result = self.client.simGetImage(0, AirSimImageType.Depth)
        if (result != "\0"):
            # depth
            rawImage = np.fromstring(result, np.int8)
            png = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
            if png is not None:
                # return pic, only first channel is enough for depth
                # apply threshold
                # png[:,:,0] = self.tsh_distance(100,png[:,:,0])
                return png[:,:,0]
            else:
                print('Couldnt take one depth pic.')
                return np.zeros((144,256)) # empty picture



if __name__ == '__main__':
    # create airsim eom object
    env = AirSimEOM()

    # basic rl loop
    for j in range(2000):
        print('**** Episode {} ****'.format(j))
        observation = env.reset()
        done = 0

        for i in range(2000):

            action = env.act()  # your agent here (this takes random actions)
            observation, reward, done, info = env.step(action)

            if done:
                pos = env.client.getPosition()
                print('[*] Total Distance = {}\n[*] Total Time = {}'.format(pos[0],i))
                break
