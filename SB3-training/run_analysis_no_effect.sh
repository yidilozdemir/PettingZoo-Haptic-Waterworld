#!/bin/bash -l

# Example batch script to run a Python script in a virtual environment.


#$ -l tmpfs=10G

# Example batch script to run a Python script in a virtual environment.
#request 24 hours 
#$ -l h_rt=20:00:00

#$ -pe smp 6

# Set the name of the job.
#$ -N pettingzoo-noeffect

# Set the working directory to project directory in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/zcbtyio/Scratch/PettingZoo-Haptic-Waterworld/SB3-training

# Load python3 module - this must be the same version as loaded when creating and
# installing dependencies in the virtual environment
module load python3/3.9

# Define a local variable pointing to the project directory in your scratch space
PROJECT_DIR=/home/zcbtyio/Scratch/PettingZoo-Haptic-Waterworld/SB3-training

# Activate the virtual environment in which you installed the project dependencies
source $PROJECT_DIR/pettingzoo-sb3/bin/activate

# Run analysis script using Python in activated virtual environment passing in path to
# directory containing input data and path to directory to write outputs to
echo "Running analysis script..."
python $PROJECT_DIR/sb3_waterworld_vector.py --n_pursuers 2 --haptic_modulation_type no_effect --haptic_weight 0.5 --policy_name MlpLstmPolicy 
echo "...done."

# Copy script outputs back to scratch space under a job ID specific subdirectory
echo "Copying analysis outputs to scratch space..."
rsync -a logs/ $PROJECT_DIR/outputs_$JOB_ID/
echo "...done"