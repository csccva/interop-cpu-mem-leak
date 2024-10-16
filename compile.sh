rm *.mod *.o *.i a.ou;
CC -xhip -munsafe-fp-atomics -g -ggdb  --offload-arch=gfx90a -O3 -c cwrappers.c 
ftn -g    -eZ -D_MPIF90 -fPIC -h noomp -h flex_mp=intolerant -O0    -lsci_cray -L/opt/rocm-6.0.3/lib -lamdhip64  -lhipblas  main.f90 cwrappers.o 
