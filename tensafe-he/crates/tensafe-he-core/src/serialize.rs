//! Serialization for client-server key exchange.
//!
//! Provides byte-level serialization for:
//! - PublicKey (client → server)
//! - Ciphertext (bidirectional)
//! - RnsPoly (building block)
//!
//! Format: flat little-endian u64 arrays. Each RnsPoly is L × N × 8 bytes.

use crate::ciphertext::{Ciphertext, PublicKey};
use crate::params::Modulus;
use crate::rns::RnsPoly;

/// Serialize an RnsPoly to bytes (L × N × 8 bytes, little-endian u64).
pub fn rns_poly_to_bytes(poly: &RnsPoly) -> Vec<u8> {
    let num_limbs = poly.limbs.len();
    let n = poly.n;
    let mut bytes = Vec::with_capacity(num_limbs * n * 8);
    for l in 0..num_limbs {
        for i in 0..n {
            bytes.extend_from_slice(&poly.limbs[l][i].to_le_bytes());
        }
    }
    bytes
}

/// Deserialize an RnsPoly from bytes.
pub fn rns_poly_from_bytes(bytes: &[u8], n: usize, num_limbs: usize) -> RnsPoly {
    assert_eq!(
        bytes.len(),
        num_limbs * n * 8,
        "Expected {} bytes for RnsPoly(n={n}, L={num_limbs}), got {}",
        num_limbs * n * 8,
        bytes.len()
    );
    let mut poly = RnsPoly::zero(n, num_limbs);
    let mut offset = 0;
    for l in 0..num_limbs {
        for i in 0..n {
            poly.limbs[l][i] = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
        }
    }
    poly
}

/// Deserialize an RnsPoly from bytes with coefficient range validation.
///
/// Each deserialized coefficient is checked against its corresponding modulus:
/// `coeff < moduli[l].value`. This prevents malformed ciphertexts with out-of-range
/// coefficients from breaking Barrett reduction (which assumes inputs < q).
///
/// Returns `Err` with a descriptive message if any coefficient is out of range.
pub fn rns_poly_from_bytes_checked(
    bytes: &[u8],
    n: usize,
    moduli: &[Modulus],
) -> Result<RnsPoly, String> {
    let num_limbs = moduli.len();
    if bytes.len() != num_limbs * n * 8 {
        return Err(format!(
            "Expected {} bytes for RnsPoly(n={n}, L={num_limbs}), got {}",
            num_limbs * n * 8,
            bytes.len()
        ));
    }
    let mut poly = RnsPoly::zero(n, num_limbs);
    let mut offset = 0;
    for l in 0..num_limbs {
        let q = moduli[l].value;
        for i in 0..n {
            let coeff = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            if coeff >= q {
                return Err(format!(
                    "Coefficient out of range at limb {l}, index {i}: \
                     coeff={coeff} >= q={q}"
                ));
            }
            poly.limbs[l][i] = coeff;
            offset += 8;
        }
    }
    Ok(poly)
}

/// Serialize a PublicKey to bytes.
/// Layout: [b_bytes | a_bytes], each is L × N × 8 bytes.
pub fn pk_to_bytes(pk: &PublicKey) -> Vec<u8> {
    let mut bytes = rns_poly_to_bytes(&pk.b);
    bytes.extend(rns_poly_to_bytes(&pk.a));
    bytes
}

/// Deserialize a PublicKey from bytes.
pub fn pk_from_bytes(bytes: &[u8], n: usize, num_limbs: usize) -> PublicKey {
    let poly_size = num_limbs * n * 8;
    assert_eq!(
        bytes.len(),
        2 * poly_size,
        "Expected {} bytes for PublicKey, got {}",
        2 * poly_size,
        bytes.len()
    );
    let b = rns_poly_from_bytes(&bytes[..poly_size], n, num_limbs);
    let a = rns_poly_from_bytes(&bytes[poly_size..], n, num_limbs);
    PublicKey { b, a }
}

/// Serialize a Ciphertext to bytes.
/// Layout: [scale_bytes (8) | c0_bytes | c1_bytes].
pub fn ct_to_bytes(ct: &Ciphertext) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&ct.scale.to_le_bytes());
    bytes.extend(rns_poly_to_bytes(&ct.c0));
    bytes.extend(rns_poly_to_bytes(&ct.c1));
    bytes
}

/// Deserialize a Ciphertext from bytes (unchecked — no coefficient range validation).
///
/// Prefer [`ct_from_bytes_checked`] for untrusted input to prevent out-of-range
/// coefficients from breaking Barrett reduction.
pub fn ct_from_bytes(bytes: &[u8], n: usize, num_limbs: usize) -> Ciphertext {
    let poly_size = num_limbs * n * 8;
    assert_eq!(
        bytes.len(),
        8 + 2 * poly_size,
        "Expected {} bytes for Ciphertext, got {}",
        8 + 2 * poly_size,
        bytes.len()
    );
    let scale = f64::from_le_bytes(bytes[..8].try_into().unwrap());
    let c0 = rns_poly_from_bytes(&bytes[8..8 + poly_size], n, num_limbs);
    let c1 = rns_poly_from_bytes(&bytes[8 + poly_size..], n, num_limbs);
    Ciphertext { c0, c1, scale }
}

/// Deserialize a Ciphertext from bytes with coefficient range validation.
///
/// Each coefficient is validated to be less than its corresponding modulus q_i.
/// This is the recommended deserialization path for untrusted input (e.g.,
/// ciphertexts received over the network from clients).
///
/// Returns `Err` if the byte length is wrong or any coefficient >= q_i.
pub fn ct_from_bytes_checked(
    bytes: &[u8],
    n: usize,
    moduli: &[Modulus],
) -> Result<Ciphertext, String> {
    let num_limbs = moduli.len();
    let poly_size = num_limbs * n * 8;
    if bytes.len() != 8 + 2 * poly_size {
        return Err(format!(
            "Expected {} bytes for Ciphertext, got {}",
            8 + 2 * poly_size,
            bytes.len()
        ));
    }
    let scale = f64::from_le_bytes(bytes[..8].try_into().unwrap());
    let c0 = rns_poly_from_bytes_checked(&bytes[8..8 + poly_size], n, moduli)?;
    let c1 = rns_poly_from_bytes_checked(&bytes[8 + poly_size..], n, moduli)?;
    Ok(Ciphertext { c0, c1, scale })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ciphertext::CkksContext;
    use crate::params::CkksParams;
    use crate::rng::TenSafeRng;

    #[test]
    fn test_pk_serialize_roundtrip() {
        let params = CkksParams::for_degree(8192);
        let ctx = CkksContext::new(params.clone());
        let mut rng = TenSafeRng::from_seed(42);
        let sk = ctx.keygen(&mut rng);
        let pk = ctx.keygen_public(&sk, &mut rng);

        let bytes = pk_to_bytes(&pk);
        let pk2 = pk_from_bytes(&bytes, params.poly_degree, params.num_limbs);

        // Verify round-trip
        for l in 0..params.num_limbs {
            assert_eq!(pk.b.limbs[l], pk2.b.limbs[l]);
            assert_eq!(pk.a.limbs[l], pk2.a.limbs[l]);
        }
    }

    #[test]
    fn test_ct_serialize_roundtrip() {
        let params = CkksParams::for_degree(8192);
        let ctx = CkksContext::new(params.clone());
        let mut rng = TenSafeRng::from_seed(42);
        let sk = ctx.keygen(&mut rng);

        let z: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ct = ctx.encrypt(&z, &sk, &mut rng);

        let bytes = ct_to_bytes(&ct);
        let ct2 = ct_from_bytes(&bytes, params.poly_degree, params.num_limbs);

        assert_eq!(ct.scale, ct2.scale);
        for l in 0..params.num_limbs {
            assert_eq!(ct.c0.limbs[l], ct2.c0.limbs[l]);
            assert_eq!(ct.c1.limbs[l], ct2.c1.limbs[l]);
        }

        // Verify deserialized ct decrypts correctly
        let decoded = ctx.decrypt(&ct2, &sk);
        for i in 0..z.len() {
            let err = (decoded[i] - z[i]).abs();
            assert!(err < 1e-4, "Slot {i}: error={err}");
        }
    }

    #[test]
    fn test_ct_serialize_roundtrip_checked() {
        let params = CkksParams::for_degree(8192);
        let ctx = CkksContext::new(params.clone());
        let mut rng = TenSafeRng::from_seed(42);
        let sk = ctx.keygen(&mut rng);

        let z: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ct = ctx.encrypt(&z, &sk, &mut rng);

        let bytes = ct_to_bytes(&ct);
        let ct2 = ct_from_bytes_checked(&bytes, params.poly_degree, &params.moduli)
            .expect("Valid ciphertext should pass checked deserialization");

        assert_eq!(ct.scale, ct2.scale);
        for l in 0..params.num_limbs {
            assert_eq!(ct.c0.limbs[l], ct2.c0.limbs[l]);
            assert_eq!(ct.c1.limbs[l], ct2.c1.limbs[l]);
        }

        // Verify deserialized ct decrypts correctly
        let decoded = ctx.decrypt(&ct2, &sk);
        for i in 0..z.len() {
            let err = (decoded[i] - z[i]).abs();
            assert!(err < 1e-4, "Slot {i}: error={err}");
        }
    }

    #[test]
    fn test_checked_deserialization_rejects_out_of_range() {
        let params = CkksParams::for_degree(8192);
        let ctx = CkksContext::new(params.clone());
        let mut rng = TenSafeRng::from_seed(42);
        let sk = ctx.keygen(&mut rng);

        let z: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ct = ctx.encrypt(&z, &sk, &mut rng);

        let mut bytes = ct_to_bytes(&ct);

        // Corrupt a coefficient in c0 (first limb, first coefficient) to be >= q
        // The scale takes the first 8 bytes, then c0 data starts at offset 8
        let q0 = params.moduli[0].value;
        let bad_coeff = q0 + 1; // Definitely out of range
        bytes[8..16].copy_from_slice(&bad_coeff.to_le_bytes());

        let result = ct_from_bytes_checked(&bytes, params.poly_degree, &params.moduli);
        assert!(
            result.is_err(),
            "Checked deserialization should reject out-of-range coefficients"
        );
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("out of range"),
            "Error message should mention 'out of range': {err_msg}"
        );
    }

    #[test]
    fn test_rns_poly_checked_valid() {
        let params = CkksParams::for_degree(8192);
        let n = 4;
        let moduli = &params.moduli[..2];
        let mut poly = RnsPoly::zero(n, 2);
        poly.limbs[0] = vec![1, 2, 3, 4];
        poly.limbs[1] = vec![10, 20, 30, 40];

        let bytes = rns_poly_to_bytes(&poly);
        let poly2 = rns_poly_from_bytes_checked(&bytes, n, moduli)
            .expect("Valid polynomial should pass checked deserialization");
        assert_eq!(poly.limbs[0], poly2.limbs[0]);
        assert_eq!(poly.limbs[1], poly2.limbs[1]);
    }

    #[test]
    fn test_rns_poly_checked_rejects_bad_coeff() {
        let params = CkksParams::for_degree(8192);
        let n = 4;
        let moduli = &params.moduli[..2];
        let mut poly = RnsPoly::zero(n, 2);
        poly.limbs[0] = vec![1, 2, 3, 4];
        poly.limbs[1] = vec![10, 20, 30, 40];

        let mut bytes = rns_poly_to_bytes(&poly);
        // Corrupt first coefficient of first limb to exceed modulus
        let q0 = moduli[0].value;
        bytes[0..8].copy_from_slice(&(q0 + 100).to_le_bytes());

        let result = rns_poly_from_bytes_checked(&bytes, n, moduli);
        assert!(result.is_err(), "Should reject coefficient >= q");
    }
}
