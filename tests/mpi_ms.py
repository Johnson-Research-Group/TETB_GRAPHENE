from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define simulation parameters
n_atoms = 100  # Total number of atoms
n_steps = 100  # Number of simulation steps
temperature = 300  # Temperature (K)
delta_t = 0.001  # Time step (arbitrary units)
k_spring = 0.1  # Spring constant (arbitrary units)

# Generate initial atomic positions
np.random.seed(42)  # For reproducibility
initial_positions = np.random.rand(n_atoms, 3)

# Divide atoms among processes
atoms_per_process = n_atoms // size
my_atoms = initial_positions[rank * atoms_per_process: (rank + 1) * atoms_per_process]

# Function to calculate forces (simplified harmonic potential)
def calculate_forces(atoms):
    displacements = atoms - atoms[:, np.newaxis]
    distances = np.linalg.norm(displacements, axis=-1)
    forces = -k_spring * displacements #/ distances[:, :, np.newaxis]  # Hooke's Law
    forces[np.isnan(forces)] = 0  # Handle division by zero
    return np.sum(forces, axis=1)

# Main molecular statics loop
for step in range(n_steps):
    print(step)
    # Calculate forces in parallel
    my_forces = calculate_forces(my_atoms)
    
    # Gather forces from all processes
    all_forces = comm.gather(my_forces, root=0)
    
    if rank == 0:
        # Process 0 updates positions using the summed forces
        total_forces = np.sum(all_forces, axis=0)
        my_atoms += total_forces * delta_t / temperature
        
        # Broadcast updated positions to all processes
        comm.bcast(my_atoms, root=0)
    
# Synchronize processes
comm.barrier()

# Print final atomic positions (only on process 0)
if rank == 0:
    print("Final Atomic Positions (Process 0):")
    print(my_atoms)

# Finalize MPI
MPI.Finalize()

