#!/bin/bash -l

# Example batch script to run a Python script in a virtual environment.

# Request 1 minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:1:0

# Request 1 gigabyte of RAM for each core/thread 
# (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Request 1 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=1G

# Set the name of the job.
#$ -N python-analysis-example

# Request 1 cores.
#$ -pe smp 1

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
python $PROJECT_DIR/sb3_waterworld_vector.py --n_pursuers 2 --haptic_modulation_type average --haptic_weight 0.5 --policy_name MlpLstmPolicy >> output_average_haptic_weight0.5.txt
echo "...done."

# Copy script outputs back to scratch space under a job ID specific subdirectory
echo "Copying analysis outputs to scratch space..."
rsync -a logs/ $PROJECT_DIR/outputs_$JOB_ID/
echo "...done"