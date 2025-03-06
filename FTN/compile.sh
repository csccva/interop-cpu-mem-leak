#module load LUMI
#module load partition/G
#module use  /appl/local/containers/test-modules  
#module load rocm/6.2.2

rm *.mod *.o *.i a.out;
hipcc -munsafe-fp-atomics -g -ggdb  --offload-arch=gfx90a -O3 -c cwrappers.cu
ftn -g    -eZ -D_MPIF90 -fPIC -h noomp -h flex_mp=intolerant -O0    -lsci_cray -L/pfs/lustrep3/scratch/project_462000394/amd-sw/rocm/rocm-6.2.2/lib -lamdhip64  -lhipblas  main.f90 cwrappers.o 

# srun -p dev-g --gpus-per-node=1  --ntasks-per-node=1 --nodes=1 -c 7     --time=01:00:00 --account=project_462000007 --mem-per-cpu=80M ./a.out 
