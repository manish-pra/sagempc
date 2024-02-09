# Simulation details
**Directly, see the config files for parameter values, e.g., params for SageMPC with car dynamics for goal-oriented safe exploration is [here](https://github.com/manish-pra/sagempc/blob/main/params/params_cluttered_car.yaml).**

This file provides details of the simulation study along with links to exact parameters.

## Environment generation
1. **Goal-directed safe exploration**: The environment folders are [here](https://github.com/manish-pra/sagempc/tree/main/experiments/goal_directed_envs). It contains 10 folders ranging from 0-9 and each folder has the following structure: 

```python
    .
    ├── env_0
    │   ├── env.png                 # Image of the env along with start(red) and end(green), e.g., [image]()
    │   ├── params_env.yaml         # Specify kernel used to sample env; start, goal position
    │   ├── env_data.pkl            # contained env sample
    ├── env1                        # Each env follows this structure
    └── ...
```

2. **Maximum domain safe exploration**: The environment folders are [here](https://github.com/manish-pra/sagempc/tree/main/experiments/safe_exploration_envs). 

3. **Environments used with car dynamics**: The environment folders are [here](https://github.com/manish-pra/sagempc/tree/main/experiments/cluttered_envs).

## Robot configuration, objective and algorithm's variant

The parameter files in the [params](https://github.com/manish-pra/sagempc/tree/main/params) folder can be used to set dynamics and objectives as follows: 

```
params["agent"]["dynamics] = 'unicycle'    (Robot dynamics: "unicycle", "bicycle", "NH_intergrator")
params["algo"]["objective] = 'SE'          (Experiment objective: "SE", "GO")
params["algo"]["type] = "MPC_V0"           (Algo: "ret", "ret_expander", "MPC_expander", "MPC", "MPC_Xn", "MPC_V0")
```

Here, "ret" represents the SEOCP, ret_expander represents the SageOC-Lq, MPC represents the SageMPC-Lq, MPC_expander represents the SageMPC-Lq, MPC_Xn represents the SageMPC with growing X_n, and MPC_V0 represents the SageMPC with growing X_n and steady state defined with velocity = 0.

Link to all the parameter values: 
1. [SageMPC with unicycle for maximum domain safe exploration](https://github.com/manish-pra/sagempc/blob/main/params/SE/params_SageMPC_unicycle.yaml)
1. [SageMPC with unicycle for goal-directed safe exploration](https://github.com/manish-pra/sagempc/blob/main/params/GO/params_SageMPC_unicycle.yaml)
1. [SageMPC with car dynamics for goal-oriented safe exploration](https://github.com/manish-pra/sagempc/blob/main/params/params_cluttered_car.yaml).

From these parameter files, the SageOC and SageMPC-Lq can be obtained by setting the algo type to "ret" and "MPC_expander" respectively. For ease of running, we still provide the corresponding parameter files in the params folders. For dynamics model (unicycle and car) parameters see [model.py](https://github.com/manish-pra/sagempc/blob/main/src/utils/model.py) directly. In our experiments, we directly provide $x^{g,o}_{n}$ using a discrete path on the underlying graph, alternatively, it can also be obtained by solving for another optimization problem in the optimistic set (implemented as Oracle class in our code).

<!-- 1. [SageOC with unicycle for maximum domain safe exploration]()
1. [SageOC with unicycle for goal-directed safe exploration]() -->

 

<!-- 1.  -->



<!-- ## Plots and accomodating data, 
1. Further, you can use the consolidate_data.py script and the plotting scripts in the apps folder to plot the results. Currently, their path is set to the pre-trained data folder. -->

<!-- 1. 
Config files: 
Parameter file for SEOCP, SageMPC, SageMPC-Lq 
Link to environment file for the env used all 10 env; for goal oriented and for safe exploration

## Car synamics
SageMPC for clutttered
SageMPC for obstacle
sagempc for goal change -->
