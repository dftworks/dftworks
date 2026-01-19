"""
Crystal Structure for DFT Calculations.

Defines a crystal structure with lattice vectors, atomic positions,
and atomic species for use in plane-wave DFT calculations.
"""

import numpy as np
from .lattice import Lattice
from .pseudopotential import AtomicSpecies


class Crystal:
    """
    Crystal structure with lattice and atoms.
    
    Attributes:
        lattice: Lattice object with lattice vectors
        atoms: List of Atom objects
        natoms: Number of atoms
        species: List of unique species
    """
    
    def __init__(self, lattice, atoms=None):
        """
        Initialize crystal structure.
        
        Args:
            lattice: Lattice object or 3x3 array of lattice vectors
            atoms: List of (symbol, position) tuples or Atom objects
        """
        if isinstance(lattice, Lattice):
            self.lattice = lattice
        else:
            self.lattice = Lattice(np.array(lattice))
        
        self.atoms = []
        self.species = {}
        
        if atoms is not None:
            for atom in atoms:
                if isinstance(atom, tuple):
                    symbol, position = atom[0], atom[1]
                    zion = atom[2] if len(atom) > 2 else self._default_zion(symbol)
                    self.add_atom(symbol, position, zion)
                else:
                    self.atoms.append(atom)
    
    def _default_zion(self, symbol):
        """Default ionic charges for common elements."""
        defaults = {
            'H': 1, 'He': 2,
            'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
            'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
            'K': 1, 'Ca': 2,
        }
        return defaults.get(symbol, 1)
    
    def add_atom(self, symbol, position, zion=None):
        """
        Add an atom to the crystal.
        
        Args:
            symbol: Element symbol
            position: Fractional coordinates [x, y, z]
            zion: Ionic charge (optional, defaults based on element)
        """
        if zion is None:
            zion = self._default_zion(symbol)
        
        # Create or get atomic species
        if symbol not in self.species:
            self.species[symbol] = AtomicSpecies(symbol, zion)
        
        atom = Atom(symbol, np.array(position), self.species[symbol])
        self.atoms.append(atom)
    
    @property
    def natoms(self):
        """Number of atoms."""
        return len(self.atoms)
    
    @property
    def volume(self):
        """Cell volume in Bohr^3."""
        return self.lattice.volume
    
    def get_positions(self):
        """Get fractional positions as array, shape (natoms, 3)."""
        return np.array([atom.position for atom in self.atoms])
    
    def get_positions_cart(self):
        """Get Cartesian positions as array, shape (natoms, 3)."""
        positions = []
        for atom in self.atoms:
            r = self.lattice.frac_to_cart(atom.position)
            positions.append(r)
        return np.array(positions)
    
    def get_symbols(self):
        """Get list of element symbols."""
        return [atom.symbol for atom in self.atoms]
    
    def get_charges(self):
        """Get ionic charges as array."""
        return np.array([atom.species.zion for atom in self.atoms])
    
    def get_species_list(self):
        """Get list of AtomicSpecies for each atom."""
        return [atom.species for atom in self.atoms]
    
    def get_unique_species(self):
        """Get list of unique species symbols."""
        return list(self.species.keys())
    
    def get_total_charge(self):
        """Total ionic charge (nuclear charge)."""
        return sum(atom.species.zion for atom in self.atoms)
    
    def display(self):
        """Print crystal information."""
        print("\n" + "=" * 60)
        print("Crystal Structure")
        print("=" * 60)
        
        print(f"\nLattice vectors (Bohr):")
        for i, v in enumerate(self.lattice.vectors):
            print(f"  a{i+1} = [{v[0]:12.6f}, {v[1]:12.6f}, {v[2]:12.6f}]")
        
        print(f"\nVolume: {self.volume:.4f} Bohr^3")
        print(f"Number of atoms: {self.natoms}")
        print(f"Species: {self.get_unique_species()}")
        
        print(f"\nAtoms (fractional coordinates):")
        print(f"  {'#':>3} {'Symbol':>6} {'x':>12} {'y':>12} {'z':>12} {'Z':>6}")
        print("  " + "-" * 54)
        
        for i, atom in enumerate(self.atoms):
            pos = atom.position
            print(f"  {i+1:3d} {atom.symbol:>6} {pos[0]:12.6f} {pos[1]:12.6f} "
                  f"{pos[2]:12.6f} {atom.species.zion:6.1f}")
    
    @staticmethod
    def cubic(a, atoms=None):
        """Create simple cubic crystal."""
        lattice = Lattice.cubic(a)
        return Crystal(lattice, atoms)
    
    @staticmethod
    def fcc(a, atoms=None):
        """Create FCC crystal."""
        lattice = Lattice.fcc(a)
        return Crystal(lattice, atoms)
    
    @staticmethod
    def bcc(a, atoms=None):
        """Create BCC crystal."""
        lattice = Lattice.bcc(a)
        return Crystal(lattice, atoms)
    
    @staticmethod
    def diamond_si():
        """Create diamond Silicon structure."""
        a = 10.263  # Si lattice constant in Bohr
        lattice = Lattice.fcc(a)
        
        crystal = Crystal(lattice)
        # Diamond has two atoms per primitive cell
        crystal.add_atom('Si', [0.0, 0.0, 0.0], zion=4)
        crystal.add_atom('Si', [0.25, 0.25, 0.25], zion=4)
        
        return crystal


class Atom:
    """
    Single atom in a crystal.
    
    Attributes:
        symbol: Element symbol
        position: Fractional coordinates
        species: AtomicSpecies reference
    """
    
    def __init__(self, symbol, position, species):
        """
        Initialize atom.
        
        Args:
            symbol: Element symbol
            position: Fractional coordinates [x, y, z]
            species: AtomicSpecies object
        """
        self.symbol = symbol
        self.position = np.array(position)
        self.species = species


def read_xyz(filename, lattice):
    """
    Read crystal structure from XYZ file.
    
    Args:
        filename: Path to XYZ file
        lattice: Lattice object (XYZ has Cartesian coords, need lattice)
    
    Returns:
        Crystal object
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    natoms = int(lines[0].strip())
    # Skip comment line
    
    crystal = Crystal(lattice)
    
    for i in range(2, 2 + natoms):
        parts = lines[i].split()
        symbol = parts[0]
        
        # Cartesian coordinates
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        
        # Convert to fractional
        pos_cart = np.array([x, y, z])
        pos_frac = lattice.cart_to_frac(pos_cart)
        
        crystal.add_atom(symbol, pos_frac)
    
    return crystal


def make_supercell(crystal, nx, ny, nz):
    """
    Create a supercell by replicating the unit cell.
    
    Args:
        crystal: Original crystal
        nx, ny, nz: Number of replications in each direction
    
    Returns:
        New Crystal object with supercell
    """
    # Scale lattice vectors
    new_vectors = crystal.lattice.vectors.copy()
    new_vectors[0] *= nx
    new_vectors[1] *= ny
    new_vectors[2] *= nz
    
    new_lattice = Lattice(new_vectors)
    new_crystal = Crystal(new_lattice)
    
    # Replicate atoms
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for atom in crystal.atoms:
                    # Scale fractional position
                    new_pos = atom.position.copy()
                    new_pos[0] = (atom.position[0] + ix) / nx
                    new_pos[1] = (atom.position[1] + iy) / ny
                    new_pos[2] = (atom.position[2] + iz) / nz
                    
                    new_crystal.add_atom(atom.symbol, new_pos, atom.species.zion)
    
    return new_crystal
