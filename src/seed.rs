/// Derives an independent deterministic seed for one named stream.
///
/// The finalizer is `SplitMix64`'s bijective output permutation. Multiplication
/// by an odd constant keeps distinct sequence numbers distinct before domain
/// separation and permutation.
pub const fn derive_seed(root: u64, sequence: u64, domain: u64) -> u64 {
    let state = root ^ domain ^ sequence.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    splitmix64(state)
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivation_is_deterministic_and_domain_separated() {
        assert_eq!(derive_seed(42, 7, 11), 0x9d3b_1870_1655_846f);
        assert_ne!(derive_seed(42, 7, 11), derive_seed(42, 8, 11));
        assert_ne!(derive_seed(42, 7, 11), derive_seed(42, 7, 12));
    }
}
