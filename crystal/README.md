# Crystal Module

The `crystal` module provides functionality for reading, storing, and manipulating crystal structures for density functional theory (DFT) calculations. It handles lattice vectors, atomic positions (in fractional coordinates), species information, and cell optimization masks.

## Features

- Read crystal structures from file format
- Store atomic positions in fractional coordinates
- Convert between fractional and Cartesian coordinates
- Group atoms by species for efficient lookup
- Support for cell optimization masks
- Calculate total valence electrons
- Output crystal structures to file

## File Format

The crystal structure file format (`in.crystal`) consists of:

**Line 0:** Scale factors (in Angstroms)
```
scale_a scale_b scale_c
```

**Line 1:** Lattice vector a and cell mask
```
a_x a_y a_z mask_a_x mask_a_y mask_a_z
```

**Line 2:** Lattice vector b and cell mask
```
b_x b_y b_z mask_b_x mask_b_y mask_b_z
```

**Line 3:** Lattice vector c and cell mask
```
c_x c_y c_z mask_c_x mask_c_y mask_c_z
```

**Lines 4+:** Atomic positions (one per line)
```
species x y z
```

Where:
- `scale_a`, `scale_b`, `scale_c`: Scale factors in Angstroms
- `a_x`, `a_y`, `a_z`: Components of lattice vector a (before scaling)
- `mask_*`: Cell mask values - "T" or "t" = 1.0 (allowed to vary), "F" or "f" = 0.0 (fixed)
- `species`: Atomic species label (e.g., "Si", "Si1", "O")
- `x`, `y`, `z`: Fractional coordinates (0-1 range typically, but can be outside for periodic systems)

### Example File

```
5.43 5.43 5.43
0.01 0.5 0.5 T T T
0.5 0.0 0.5 T T T 
0.5 0.5 0.0 T T T
Si1 -0.125 -0.122 -0.126
Si2  0.122  0.123  0.125
Si2  0.0  0.123  0.125
```

## Usage

### Reading a Crystal Structure

```rust
use crystal::Crystal;

let mut crystal = Crystal::new();
crystal.read_file("in.crystal");
```

### Accessing Information

```rust
// Get number of atoms
let n_atoms = crystal.get_n_atoms();

// Get unique species
let species = crystal.get_unique_species();
let n_species = crystal.get_n_unique_species();

// Get atomic positions (fractional coordinates)
let positions = crystal.get_atom_positions();

// Get atomic positions (Cartesian coordinates in Bohr)
let positions_cart = crystal.get_atom_positions_cart();

// Get atom indices for a specific species
let si_indices = crystal.get_atom_indices_of_specie(0);

// Get positions for a specific species
let si_positions = crystal.get_atom_positions_of_specie(0);
```

### Coordinate Conversion

```rust
// Get Cartesian positions
let cart_positions = crystal.get_atom_positions_cart();

// Set positions from Cartesian coordinates
crystal.set_atom_positions_from_cart(&cart_positions);

// Set positions from fractional coordinates
crystal.set_atom_positions_from_frac(&frac_positions);
```

### Lattice Operations

```rust
// Get lattice structure
let latt = crystal.get_latt();

// Update lattice vectors
let new_latt = Lattice::new(&vec_a, &vec_b, &vec_c);
crystal.set_lattice_vectors(&new_latt);

// Get cell mask (for optimization)
let mask = crystal.get_cell_mask();
```

### Output

```rust
// Display crystal structure to stdout
crystal.display();

// Write crystal structure to file
crystal.output();  // Creates "out.crystal"
```

## API Reference

### Main Methods

- `new()` - Create a new empty crystal structure
- `read_file(path)` - Read crystal structure from file
- `output()` - Write crystal structure to "out.crystal"
- `display()` - Print formatted crystal structure information

### Getters

- `get_n_atoms()` - Total number of atoms
- `get_n_unique_species()` - Number of unique species
- `get_unique_species()` - List of unique species
- `get_atom_positions()` - Atomic positions (fractional)
- `get_atom_positions_cart()` - Atomic positions (Cartesian, in Bohr)
- `get_atom_positions_of_specie(isp)` - Positions for species `isp`
- `get_atom_indices_of_specie(isp)` - Atom indices for species `isp`
- `get_atom_species()` - Species labels for all atoms
- `get_atom_types()` - Type indices (1-based) for all atoms
- `get_latt()` - Lattice structure
- `get_cell_mask()` - Cell optimization mask matrix

### Setters

- `set_atom_positions_from_cart(positions)` - Set positions from Cartesian coordinates
- `set_atom_positions_from_frac(positions)` - Set positions from fractional coordinates
- `set_lattice_vectors(latt)` - Update lattice vectors

### Calculations

- `get_n_total_electrons(pots)` - Calculate total valence electrons
- `get_zions(pots)` - Get valence charges for all atoms

## Coordinate Systems

The crystal module uses two coordinate systems:

1. **Fractional coordinates**: Stored internally, relative to lattice vectors (0-1 range typically)
2. **Cartesian coordinates**: Absolute positions in Bohr (converted on demand)

Lattice vectors are stored in Bohr internally, but the input file format uses Angstroms with scale factors.

## Testing

Run tests using the justfile:

```bash
just test              # Run all tests
just test-verbose      # Run tests with output
just test-serial       # Run tests sequentially
just test-specific <test_name>  # Run a specific test
```

Or directly with cargo:

```bash
cargo test --package crystal
```

## Dependencies

- `lattice` - Lattice vector operations
- `vector3` - 3D vector operations
- `matrix` - Matrix operations
- `pspot` - Pseudopotential handling
- `dwconsts` - Physical constants
- `itertools` - Iterator utilities

## Notes

- Atomic positions are stored in fractional coordinates internally
- Lattice vectors are converted from Angstroms to Bohr on input
- The cell mask indicates which lattice components can vary during optimization
- Species grouping is optimized for O(n) lookup using HashMap
- Empty lines in input files are automatically skipped
