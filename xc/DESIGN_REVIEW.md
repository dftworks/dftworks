# XC Module Design Review

## Current Design Overview

The `xc` module implements exchange-correlation functionals for DFT calculations. It uses:
- A trait `XC` with a single method `potential_and_energy`
- Factory function `new()` returning `Box<dyn XC>` based on string matching
- Three implementations: `XCLDAPZ`, `XCLSDAPZ`, `XCPBE` (stub)

## Issues Identified

### 1. **Code Duplication**
- `evc_pz` (unpolarized) and `evc_pz_u`/`evc_pz_p` (polarized) share nearly identical logic
- The PZ correlation parameters differ only by values, not structure
- Duplication makes maintenance harder and increases bug risk

### 2. **String-Based Factory Function**
- Not type-safe: typos cause runtime panics
- No compile-time checking
- Hard to discover available schemes
- Error handling via `panic!()` is not user-friendly

### 3. **Trait Objects Performance Overhead**
- `Box<dyn XC>` has dynamic dispatch overhead
- Virtual function calls in hot loops (called for every grid point)
- Could use zero-cost abstractions (enums) instead

### 4. **No Error Handling**
- Panics on unknown XC schemes
- No way to handle errors gracefully
- Should return `Result` types

### 5. **Empty Structs**
- `XCLDAPZ {}`, `XCLSDAPZ {}`, `XCPBE {}` are zero-sized
- Could be unit structs or better yet, an enum
- No configuration or state possible

### 6. **Poor Separation of Concerns**
- Exchange and correlation are tightly coupled
- Can't reuse exchange without correlation or vice versa
- Hard to compose different exchange/correlation combinations

### 7. **Module Organization**
- All functionals in separate files but no clear hierarchy
- Common code (PZ correlation) not shared
- No clear separation between exchange, correlation, and functional types

### 8. **Missing Features**
- No way to query functional properties (needs gradient? spin-polarized?)
- No validation of input/output shapes
- PBE implementation is empty stub

## Recommended Improvements

### Option A: Enum-Based Design (Recommended for Performance)

```rust
// Type-safe, zero-cost abstraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XCFunctional {
    LdaPz,
    LsdaPz,
    Pbe,
}

impl XCFunctional {
    pub fn from_str(s: &str) -> Result<Self, XCError> {
        match s {
            "lda-pz" => Ok(XCFunctional::LdaPz),
            "lsda-pz" => Ok(XCFunctional::LsdaPz),
            "pbe" => Ok(XCFunctional::Pbe),
            _ => Err(XCError::UnknownScheme(s.to_string())),
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            XCFunctional::LdaPz => "lda-pz",
            XCFunctional::LsdaPz => "lsda-pz",
            XCFunctional::Pbe => "pbe",
        }
    }
}

impl XC for XCFunctional {
    fn potential_and_energy(...) {
        match self {
            XCFunctional::LdaPz => ldapz::compute(...),
            XCFunctional::LsdaPz => lsdapz::compute(...),
            XCFunctional::Pbe => pbe::compute(...),
        }
    }
}
```

**Benefits:**
- Zero-cost abstraction (no heap allocation, no dynamic dispatch)
- Type-safe at compile time
- Better performance in hot loops
- Easy to extend with new functionals

### Option B: Composable Exchange/Correlation Design

```rust
// Separate exchange and correlation traits
pub trait Exchange {
    fn exchange(&self, rho: f64, zeta: Option<f64>) -> ExchangeResult;
}

pub trait Correlation {
    fn correlation(&self, rho: f64, zeta: Option<f64>) -> CorrelationResult;
}

// Compose them
pub struct XCFunctional {
    exchange: Box<dyn Exchange>,
    correlation: Box<dyn Correlation>,
}

impl XC for XCFunctional {
    fn potential_and_energy(...) {
        let ex_result = self.exchange.exchange(...);
        let ec_result = self.correlation.correlation(...);
        // Combine results
    }
}
```

**Benefits:**
- Maximum flexibility: mix any exchange with any correlation
- Better code reuse
- Easier to test components independently
- More modular

### Option C: Hybrid Approach (Best of Both Worlds)

```rust
// Common exchange/correlation implementations
mod exchange {
    pub fn slater(rho: f64) -> (f64, f64) { ... }
    pub fn slater_spin(rho: f64, zeta: f64) -> (f64, f64, f64) { ... }
}

mod correlation {
    pub fn pz(rho: f64, spin: Spin) -> (f64, f64) { ... }
    // Shared implementation for unpolarized and polarized
}

// Functional enum for type safety and performance
#[derive(Debug, Clone, Copy)]
pub enum XCFunctional {
    LdaPz,
    LsdaPz,
    Pbe,
}

impl XCFunctional {
    pub fn compute(&self, rho: &RHOR, ...) {
        match self {
            XCFunctional::LdaPz => {
                let (vx, ex) = exchange::slater(rho);
                let (vc, ec) = correlation::pz(rho, Spin::Unpolarized);
                // combine
            }
            // ...
        }
    }
}
```

## Specific Recommendations

### 1. Extract Common PZ Correlation Code

Create a shared `correlation::pz` module:

```rust
// xc/src/correlation/pz.rs
pub struct PZCorrelation {
    params: PZParams,
}

pub enum PZParams {
    Unpolarized,  // Uses evc_pz parameters
    Polarized,    // Uses evc_pz_u and evc_pz_p parameters
}

impl PZCorrelation {
    pub fn unpolarized() -> Self { ... }
    pub fn polarized() -> Self { ... }
    
    pub fn compute(&self, rho: f64, zeta: Option<f64>) -> (f64, f64) {
        // Single implementation handling both cases
    }
}
```

### 2. Better Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum XCError {
    #[error("Unknown XC scheme: {0}")]
    UnknownScheme(String),
    #[error("Invalid density: {0}")]
    InvalidDensity(String),
    #[error("Unsupported spin configuration")]
    UnsupportedSpin,
}
```

### 3. Add Functional Metadata

```rust
pub trait XC {
    fn potential_and_energy(...);
    
    // Query functional properties
    fn needs_gradient(&self) -> bool;
    fn supports_spin(&self) -> bool;
    fn name(&self) -> &'static str;
}
```

### 4. Module Reorganization

```
xc/
├── src/
│   ├── lib.rs           # Public API, trait definitions
│   ├── functional.rs    # XCFunctional enum
│   ├── exchange/
│   │   ├── mod.rs
│   │   ├── slater.rs
│   │   └── pbe.rs
│   ├── correlation/
│   │   ├── mod.rs
│   │   ├── pz.rs        # Shared PZ implementation
│   │   └── pbe.rs
│   ├── lda/
│   │   ├── mod.rs
│   │   ├── ldapz.rs     # Uses exchange::slater + correlation::pz
│   │   └── lsdapz.rs
│   └── gga/
│       └── pbe.rs
```

### 5. Performance Considerations

- Use `#[inline]` on hot path functions
- Consider SIMD for vectorized operations (rho^(1/3) calculations)
- Pre-compute constants where possible
- Use iterator chains for better vectorization (see performance guidelines)

### 6. Testing

Add unit tests for:
- Each exchange/correlation component
- Edge cases (very low/high density)
- Spin polarization handling
- Numerical accuracy against reference values

## Migration Path

1. **Phase 1**: Extract common PZ correlation code, keep trait-based API
2. **Phase 2**: Add enum-based `XCFunctional`, keep both APIs
3. **Phase 3**: Deprecate string-based factory, migrate callers
4. **Phase 4**: Remove trait objects, use enum dispatch

## Conclusion

The current design works but has room for improvement in:
- **Performance**: Switch from trait objects to enums
- **Type Safety**: Replace strings with enums
- **Code Reuse**: Extract common correlation code
- **Error Handling**: Use Result types instead of panics
- **Modularity**: Separate exchange and correlation concerns

The hybrid approach (Option C) provides the best balance of performance, type safety, and code reuse while maintaining backward compatibility during migration.
