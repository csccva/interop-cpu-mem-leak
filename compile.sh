module load LUMI
module load partition/G
module load rocm

rm *.mod *.o *.i a.out;
CC -xhip -munsafe-fp-atomics -g -ggdb  --offload-arch=gfx90a -O3 -c cwrappers.c 
ftn -g    -eZ -D_MPIF90 -fPIC -h noomp -h flex_mp=intolerant -O0    -lsci_cray -L/opt/rocm-6.0.3/lib -lamdhip64  -lhipblas  main.f90 cwrappers.o 

# srun -p dev-g --gpus-per-node=1  --ntasks-per-node=1 --nodes=1 -c 7     --time=01:00:00 --account=project_462000007 --mem-per-cpu=750M ./a.out 
