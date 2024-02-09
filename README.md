# SageMPC

<!-- This repository contains the code for the paper "Safe guaranteed exploration for non-linear systems". The paper is available on [arXiv](https://arxiv.org/abs/2109.14889).  -->
<!-- The code is based on the [acados]() library and the [casadi]() library. The code is written in Python 3.7 and tested on Ubuntu 18.04.    -->

This repository contains code for the "Safe guaranteed exploration for non-linear systems" work. The simulation details for the paper can be found [here](https://github.com/manish-pra/sagempc/blob/main/docs/simulation_details.md). 

## Getting started

1. Clone the repository and install the dependencies, especially, [acados](https://docs.acados.org/installation/), [casadi](https://web.casadi.org/get/) and check [requirements.txt](https://github.com/manish-pra/sagempc/blob/main/requirements.txt).


1. To run the code, use the following command

    ```
    python3 sagempc/main.py -i $i -env $env_i -param $param_i
    ```
    where,

    ```
    $param_i: Name of the param file (see params folder) to pick an algo and the env type 
    $env_i  : An integer to pick an instance of the environment
    $i      : An integer to run multiple instances
    ```

    E.g., the following command runs SageMPC on the cluttered environment with env_0 and i=2 instance

    ``` 
    python3 sagempc/main.py -i 2 -env 0 -param "params_cluttered_car"
    ```

1. For visualizations/videos use the following script once your experiment is completed

    ```
    python3 sagempc/video.py
    ```

<!-- 4. Each env folder contains an [image]() of the environment. If turned on, i.e., "params["visu"]["show"] = True", each experiment will produce a .mp4 along with running, however, it could take up some time so unless used for debugging we suggest keeping it off and generate visualization later with "video.py"

5. Shell script to run multiple env in parallel

6. Consolidate and Plotting -->

## Simulation videos
<p align="center"><strong>Goal-directed safe exploration using Car dynamics in unknown, challenging environments.</strong></p>

<img src="https://github.com/manish-pra/sagempc/blob/main/docs/gifs/SageMPC_car_cluttered.gif" width="400" height="350">  <img src="https://github.com/manish-pra/sagempc/blob/main/docs/gifs/SageMPC_car_obstacle.gif" width="420" height="350">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;a) A cluttered environment &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) Environment with a large obstable

<p align="justify">
The car, depicted by the brown box, starts safely at the red star and needs to navigate through a priori unknown obstacles represented by the black region to minimize the loss function, shown by contours, which captures the task of reaching the goal location marked by the green star. Note that we see the black obstacle but the car doesn't know about it and needs to learn about it from samples. To explore the region, it collects measurements depicted by the red points and gradually grows its reachable returnable pessimistic set, shown by the black-yellow lines. Throughout the process, the car does not violate any of the safety-critical constraints, and the resulting safe trajectory traversed by the car is depicted by the blue line. </p>

Similar video for the non-reachable goal environment (Fig. 12 c) is [here](https://github.com/manish-pra/sagempc/blob/main/docs/gifs/SageMPC_car_non_reachable_goal.gif). 





<!-- ## Citing 
1. add comments in the config file, we directly find x^{g,0} using discrete path, model params
3. add config params in other config file
4. test all files one and then commit
6. check code once and clean up the easily doable things -->
