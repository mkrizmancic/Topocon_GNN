#PBS -N GNN_l2_approx
#PBS -q gpu
#PBS -o /lustre/home/mkrizman/Topocon_GNN/gnn_fiedler_approx/HPC/jobs/output/
#PBS -e /lustre/home/mkrizman/Topocon_GNN/gnn_fiedler_approx/HPC/jobs/error/
#PBS -l select=1:ncpus=4:ngpus=1:mem=16GB

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

cd $HOME/Topocon_GNN/gnn_fiedler_approx

apptainer run --nv HPC/gnn_fiedler.sif wandb agent $sweep_ID
