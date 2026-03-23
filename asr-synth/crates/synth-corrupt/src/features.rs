/// Phoneme feature-based substitution cost.
///
/// Returns a cost between 0.0 (identical) and 1.0 (maximally different).
/// Phonemes that share place, manner, or voicing are cheaper to substitute.
pub fn substitution_cost(a: &str, b: &str) -> f32 {
    if a == b {
        return 0.0;
    }

    let fa = features(a);
    let fb = features(b);

    // If either is unknown, use max cost
    if fa.is_none() || fb.is_none() {
        return 1.0;
    }
    let fa = fa.unwrap();
    let fb = fb.unwrap();

    // Vowel-vowel or consonant-consonant?
    if fa.is_vowel != fb.is_vowel {
        return 1.0; // vowel↔consonant is maximally different
    }

    if fa.is_vowel {
        vowel_distance(&fa, &fb)
    } else {
        consonant_distance(&fa, &fb)
    }
}

#[derive(Clone, Copy)]
struct PhonemeFeatures {
    is_vowel: bool,
    // Consonant features
    place: u8,    // 0=bilabial 1=labiodental 2=dental 3=alveolar 4=postalveolar 5=velar 6=glottal
    manner: u8,   // 0=stop 1=fricative 2=affricate 3=nasal 4=liquid 5=glide
    voiced: bool,
    // Vowel features
    height: u8,   // 0=high 1=mid 2=low
    backness: u8, // 0=front 1=central 2=back
    rounded: bool,
}

fn consonant_distance(a: &PhonemeFeatures, b: &PhonemeFeatures) -> f32 {
    let mut cost = 0.0f32;

    // Voicing difference: small cost (0.2)
    if a.voiced != b.voiced {
        cost += 0.2;
    }

    // Place difference: scaled by distance
    let place_diff = (a.place as i8 - b.place as i8).unsigned_abs();
    cost += match place_diff {
        0 => 0.0,
        1 => 0.2, // adjacent places (alveolar↔postalveolar)
        2 => 0.4,
        _ => 0.6,
    };

    // Manner difference: scaled by distance
    let manner_diff = (a.manner as i8 - b.manner as i8).unsigned_abs();
    cost += match manner_diff {
        0 => 0.0,
        1 => 0.2, // stop↔fricative, fricative↔affricate
        _ => 0.4,
    };

    cost.min(1.0)
}

fn vowel_distance(a: &PhonemeFeatures, b: &PhonemeFeatures) -> f32 {
    let mut cost = 0.0f32;

    let height_diff = (a.height as i8 - b.height as i8).unsigned_abs();
    cost += height_diff as f32 * 0.25;

    let back_diff = (a.backness as i8 - b.backness as i8).unsigned_abs();
    cost += back_diff as f32 * 0.25;

    if a.rounded != b.rounded {
        cost += 0.15;
    }

    cost.min(1.0)
}

fn features(phoneme: &str) -> Option<PhonemeFeatures> {
    Some(match phoneme {
        // ── Consonants ──────────────────────────────────────────
        // Stops
        "P"  => PhonemeFeatures { is_vowel: false, place: 0, manner: 0, voiced: false, height: 0, backness: 0, rounded: false },
        "B"  => PhonemeFeatures { is_vowel: false, place: 0, manner: 0, voiced: true,  height: 0, backness: 0, rounded: false },
        "T"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 0, voiced: false, height: 0, backness: 0, rounded: false },
        "D"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 0, voiced: true,  height: 0, backness: 0, rounded: false },
        "K"  => PhonemeFeatures { is_vowel: false, place: 5, manner: 0, voiced: false, height: 0, backness: 0, rounded: false },
        "G"  => PhonemeFeatures { is_vowel: false, place: 5, manner: 0, voiced: true,  height: 0, backness: 0, rounded: false },
        // Fricatives
        "F"  => PhonemeFeatures { is_vowel: false, place: 1, manner: 1, voiced: false, height: 0, backness: 0, rounded: false },
        "V"  => PhonemeFeatures { is_vowel: false, place: 1, manner: 1, voiced: true,  height: 0, backness: 0, rounded: false },
        "TH" => PhonemeFeatures { is_vowel: false, place: 2, manner: 1, voiced: false, height: 0, backness: 0, rounded: false },
        "DH" => PhonemeFeatures { is_vowel: false, place: 2, manner: 1, voiced: true,  height: 0, backness: 0, rounded: false },
        "S"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 1, voiced: false, height: 0, backness: 0, rounded: false },
        "Z"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 1, voiced: true,  height: 0, backness: 0, rounded: false },
        "SH" => PhonemeFeatures { is_vowel: false, place: 4, manner: 1, voiced: false, height: 0, backness: 0, rounded: false },
        "ZH" => PhonemeFeatures { is_vowel: false, place: 4, manner: 1, voiced: true,  height: 0, backness: 0, rounded: false },
        "HH" => PhonemeFeatures { is_vowel: false, place: 6, manner: 1, voiced: false, height: 0, backness: 0, rounded: false },
        // Affricates
        "CH" => PhonemeFeatures { is_vowel: false, place: 4, manner: 2, voiced: false, height: 0, backness: 0, rounded: false },
        "JH" => PhonemeFeatures { is_vowel: false, place: 4, manner: 2, voiced: true,  height: 0, backness: 0, rounded: false },
        // Nasals
        "M"  => PhonemeFeatures { is_vowel: false, place: 0, manner: 3, voiced: true,  height: 0, backness: 0, rounded: false },
        "N"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 3, voiced: true,  height: 0, backness: 0, rounded: false },
        "NG" => PhonemeFeatures { is_vowel: false, place: 5, manner: 3, voiced: true,  height: 0, backness: 0, rounded: false },
        // Liquids
        "L"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 4, voiced: true,  height: 0, backness: 0, rounded: false },
        "R"  => PhonemeFeatures { is_vowel: false, place: 3, manner: 4, voiced: true,  height: 0, backness: 0, rounded: false },
        // Glides
        "W"  => PhonemeFeatures { is_vowel: false, place: 0, manner: 5, voiced: true,  height: 0, backness: 0, rounded: false },
        "Y"  => PhonemeFeatures { is_vowel: false, place: 4, manner: 5, voiced: true,  height: 0, backness: 0, rounded: false },

        // ── Vowels ──────────────────────────────────────────────
        //                                           height  backness  rounded
        "IY" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 0, backness: 0, rounded: false },
        "IH" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 0, backness: 0, rounded: false },
        "EH" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 0, rounded: false },
        "EY" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 0, rounded: false },
        "AE" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 2, backness: 0, rounded: false },
        "AA" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 2, backness: 2, rounded: false },
        "AO" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 2, rounded: true },
        "OW" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 2, rounded: true },
        "OY" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 2, rounded: true },
        "UH" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 0, backness: 2, rounded: true },
        "UW" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 0, backness: 2, rounded: true },
        "AH" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 1, rounded: false },
        "ER" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 1, backness: 1, rounded: false },
        "AW" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 2, backness: 1, rounded: false },
        "AY" => PhonemeFeatures { is_vowel: true, place: 0, manner: 0, voiced: true, height: 2, backness: 1, rounded: false },

        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voicing_pair_is_cheap() {
        // K↔G is just voicing
        assert!(substitution_cost("K", "G") < 0.3);
        assert!(substitution_cost("T", "D") < 0.3);
        assert!(substitution_cost("P", "B") < 0.3);
        assert!(substitution_cost("S", "Z") < 0.3);
    }

    #[test]
    fn different_place_is_expensive() {
        // K↔P is place change (velar→bilabial)
        assert!(substitution_cost("K", "P") > substitution_cost("K", "G"));
        // K↔B is place + voicing
        assert!(substitution_cost("K", "B") > substitution_cost("K", "G"));
    }

    #[test]
    fn same_phoneme_is_zero() {
        assert_eq!(substitution_cost("K", "K"), 0.0);
        assert_eq!(substitution_cost("AE", "AE"), 0.0);
    }

    #[test]
    fn vowel_consonant_is_max() {
        assert_eq!(substitution_cost("K", "AE"), 1.0);
    }

    #[test]
    fn similar_vowels_are_cheap() {
        // IY↔IH (both high front unrounded)
        assert!(substitution_cost("IY", "IH") < 0.2);
        // AE↔EH (front, adjacent height)
        assert!(substitution_cost("AE", "EH") < 0.4);
    }
}
