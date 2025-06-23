#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                        # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=llama
#SBATCH --output=/projects/klybarge/HPV_Information_Extraction/Llama/utility_r/output.out   # Output file
#SBATCH --error=/projects/klybarge/HPV_Information_Extraction/Llama/utility_r/error.err    # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12                 # number of cores needed
#SBATCH --gres=gpu:A100.80gb:4               # up to 8; only request what you need
#SBATCH --mem=100GB
#SBATCH --export=ALL
#SBATCH --time=0-10:00:00                   # set to 2hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python

# python -m venv llamaenv 
# source llamaenv/bin/activate 
#pip install faiss-gpu


# pip install torch accelerate transformers


source /projects/klybarge/HPV_Information_Extraction/Llama/Llama3_HPV_Zeroshot_w_guidelines/llamaenv/bin/activate 
#pip install PyPDF2

#pip install json
#pip install --upgrade transformers

#pip install torch==2.2.2
# pip install torch accelerate transformers json datetime numpy
#pip list


python main.py --hotpot_json data/hotpot_first200.json
#python main.py --rgb_json data/rgb.json --top_k 10

#python main.py --beerqa_json data/beer.json
#python  main.py --file data/beerqa1.txt --query "Anti-infective drugs became more prominent after what war?"

