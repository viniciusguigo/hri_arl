# GOALS
## US Army Research Laboratory (ARL) - Summer 2017

This document describes the requirements and resources for the work to be done
during the Summer Internship at US Army Research Laboratory (ARL), SEDD <ADD INFO ABOUT SEDD>  

Personnel Involved:
* Dr. William Nothwang
* Greg <LAST NAME>
* Daniel Donavanik
* Vinicius Guimaraes Goecks

Last update: May 16, 2017.  

Vinicius Guimaraes Goecks  

### Primary Requirements

1. Use Deep Reinforcement Learning and Artificial Intelligence techniques to learn how to
autonomously fly and navigate a quadcopter, equipped with sensors described at item 2,
through an unknown map and achieve a desired position or state.  
(maintain stable flight and collision avoidance)

2. The quadcopter is equipped with the following sensors:   

- RBG-D Camera;
- Onboard Image Segmentation;
- Pixhawk flight-controller: GPS, Gyro, Accelerometer, and Magnetometer.

3. After learned policy is deployed to quadcopter, only onboard processing is
allowed to be used to control the vehicle.

### Secondary Requirements

1. Use human-input to accelerate the policy learned and compare with results
obtained from the policy learned only using the onboard sensors.

2. Analyze the effects of using different camera image resolutions to achieve the
primary requirements.

3. Take a look on the VizDoom environment and see what kind of policies can be
trained there. Study future possibilities of implementing similar approaches to
the quadcopter project.

### Resources and Methods

The following methods and resources will be used to achieve the described
primary and secondary requirements:

- Microsoft AirSim simulates the quadcopter flight dynamics and the surrounding
environment with high degree of accuracy. This includes shades, lighting variation,
sun glare, collision, etc.

- Use the Advantage A... Actor-Critic (A3C) algorithm to achieve the primary
requirements using only onboard sensors (Pixhawk) and camera data provided by
the Microsoft AirSim software.

- Use <NAME IMITATION LEARNING> algorithm to speed-up the learning process and
compare the results with the policy learned by the A3C algorithm.

### Additional Info

- Take a look on the project TAMER (?) by UT Austin, in which human feedback is
used to critique the learned policies.


### End of Summer Product
- Split primary requirements in smaller steps (sort of dumbing down the problem)
to make sure some goals will be achieved during the Summer.
- Ambitious: achieve all Primary Requirements.
- More ambitious: transfer policy learned when achieving the primary requirements
to hardware and compare the results.
- Prepare publication to some AI or Robotics/Computer Vision conference. Good
candidate: AAAI 2018 in Louisiana.  
