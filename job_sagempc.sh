#!/bin/bash
module load gcc/8.2.0 python/3.11.2 ffmpeg/5.0 eth_proxy glew/2.1.0 glfw/3.3.4 mesa/18.3.6 cuda/11.4.2 cudnn/8.2.4.15 cmake/3.25.0
export ACADOS_SOURCE_DIR=/cluster/project/infk/krause/manishp/work_mpc/acados
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cluster/project/infk/krause/manishp/work_mpc/acados/lib

for env_i in {0..0}
do
   for param_i in "params_SE_unicycle_MPC_exp"
   do
      for i in {4..8}
      do
         echo "Welcome to env $env_i defined by param $param_i for $i times"
         # sbatch -n 40 --time=3:58:00 --mem-per-cpu=1024 --wrap="python3 subrl/car_racing/subrl.py -i $i -param $param_i"
         sbatch -n 4 --time=3:58:00 --mem-per-cpu=2048 --wrap="python3 safe-mpc/main.py -i $i -param $param_i -env $env_i"
         sleep 60
         # echo "Welcome to env $env_i defined by param $param_i for $i times"
         # sbatch --wrap="python3 safe-mpc/main.py -i $i -param $param_i -env $env_i"
         # python3 safe-mpc/main.py -i $i -param $param_i -env $env_i &
         # sbatch -n 40 --time=3:58:00 --mem-per-cpu=1024 --wrap="python3 subrl/car_racing/subrl.py -i $i -param $param_i"
        #  sbatch -n 40 --time=3:58:00 --mem-per-cpu=1024 --wrap="python3 safe-mpc/main.py -i $i -param $param_i -env $env_i"
      done
   done
done

# nohup ./job_sempc.sh &

# sbatch -n 4 --time=3:58:00 --mem-per-cpu=2048 --wrap="python3 safe-mpc/apps/consolidate_data.py"

# pip version change by: curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8
##"params_env_unicycle_ret"