
! ftn -g    -eZ -D_MPIF90 -fPIC -h noomp -h flex_mp=intolerant -O0    -lsci_cray -L/opt/rocm-6.0.3/lib -lamdhip64  -lhipblas main.f90 cwrappers.o
MODULE F_B_C
    INTERFACE
      subroutine gpu_malloc_all(a_d,n,gpu_stream) bind(C,name="cuda_malloc_all")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d,gpu_stream
        integer(c_size_t),value :: n
      end subroutine
      
      subroutine gpu_memset_async(a_d,valuetoset,n,gpu_stream) bind(C,name="cuda_memset_async")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
        integer(c_int),value :: valuetoset
      end subroutine
      
      subroutine gpu_device_reset() bind(C,name="cuda_device_reset")
        use iso_c_binding
        implicit none
      end subroutine

      subroutine gpu_free_async(a_d,gpu_stream) bind(C,name="cuda_free_async")
        use iso_c_binding
        implicit none
        type(c_ptr) :: a_d
        type(c_ptr) :: gpu_stream
      end subroutine

      subroutine cpy_htod(a,a_d,n, gpu_stream) bind(C,name="cuda_cpy_htod")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine

      subroutine cpy_dtod(b_d,a_d,n, gpu_stream) bind(C,name="cuda_cpy_dtod")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,b_d
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine

      subroutine cpy_dtoh(a_d,a,n, gpu_stream) bind(C,name="cuda_cpy_dtoh")
        use iso_c_binding
        implicit none
        type(c_ptr),value :: a_d,a
        type(c_ptr) :: gpu_stream
        integer(c_size_t),value :: n
      end subroutine
      
      subroutine create_cublas_handle(cubhandle, gpu_stream)bind(C,name="create_cublas_handle")
        use iso_c_binding
        implicit none
        type(c_ptr) :: cubhandle, gpu_stream
      end subroutine

      subroutine destroy_cublas_handle(cubhandle, gpu_stream)bind(C,name="destroy_cublas_handle")
        use iso_c_binding
        implicit none
        type(c_ptr) :: cubhandle,gpu_stream
      end subroutine

      subroutine gpu_set_device(my_rank) bind(C, name="cuda_set_device")
        use iso_c_binding
        integer(c_int), value :: my_rank
      end subroutine

      subroutine gpu_device_sync() bind(C,name="gpu_device_sync")
        use iso_c_binding
        implicit none
      end subroutine

      END INTERFACE
    END MODULE F_B_C

program interpolation
    use iso_c_binding
    use F_B_C
    use mpi

    implicit none
    real*8, allocatable :: A_h(:,:), B_h(:,:)
    real*8, allocatable :: C_h(:,:), D_h(:,:)
    integer :: nx=256, ny=256
    type(c_ptr) :: A_d, B_d
    type(c_ptr) :: cublas_handle, gpu_stream
    integer :: ierr, rank, ntasks
    integer :: n_ii
    integer(c_size_t) :: st_A_d,st_B_d

    allocate(A_h(1:nx,1:ny),B_h(1:nx,1:ny))
    allocate(C_h(1:nx,1:ny),D_h(1:nx,1:ny))
    
    call random_number(A_h)
    call random_number(B_h)
    
    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, ntasks, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, rank, ierr)
    
    call gpu_set_device(rank) ! This works when each GPU has only 1 visible device. This is done in the slurm submission script
    
    call create_cublas_handle(cublas_handle, gpu_stream) 
    st_A_d=nx*ny*c_double
    st_B_d=nx*ny*c_double
    
    do n_ii=1,10000000
        call gpu_malloc_all(A_d,st_A_d,gpu_stream)
        call gpu_malloc_all(B_d,st_B_d,gpu_stream)
        
        call cpy_htod(c_loc(A_h),A_d, st_A_d,gpu_stream)
        call cpy_htod(c_loc(B_h),B_d, st_B_d,gpu_stream)

        call cpy_dtoh(B_d,c_loc(D_h), st_B_d,gpu_stream)
        call cpy_dtoh(A_d,c_loc(C_h), st_A_d,gpu_stream)
    
        call gpu_free_async(A_d,gpu_stream)
        call gpu_free_async(B_d,gpu_stream)

        call gpu_device_sync()

        write(*,*) n_ii,sum(abs(A_h-C_h)), sum(abs(B_h-D_h))
    enddo

    deallocate(A_h,B_h,C_h,D_h)

  call mpi_finalize(ierr)
  
  call destroy_cublas_handle(cublas_handle, gpu_stream)

  call gpu_device_reset()

end program
