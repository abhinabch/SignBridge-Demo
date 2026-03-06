/**
 * gestureLabels.ts
 * 
 * Central registry of all supported ASL gesture labels.
 */

/** A–Z fingerspelling alphabet */
export const ALPHABET_SIGNS: string[] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
];

/** Common everyday signs */
export const COMMON_SIGNS: string[] = [
    'Hello', 'Thank You', 'Yes', 'No', 'Please',
    'Sorry', 'Help', 'More', 'Stop', 'Good',
    'Bad', 'Water', 'Eat', 'Home', 'Love',
];

/** Combined array of every supported gesture */
export const ALL_GESTURES: string[] = [...ALPHABET_SIGNS, ...COMMON_SIGNS];

/**
 * Difficulty level for each gesture.
 * Alphabet and basic greetings are Beginner; multi-motion signs are Intermediate.
 */
export const GESTURE_DIFFICULTY: Record<string, 'Beginner' | 'Intermediate'> = (() => {
    const map: Record<string, 'Beginner' | 'Intermediate'> = {};
    for (const letter of ALPHABET_SIGNS) {
        // J and Z require motion and are intermediate
        map[letter] = (letter === 'J' || letter === 'Z') ? 'Intermediate' : 'Beginner';
    }
    // Common signs: simple static ones are beginner, motion-based are intermediate
    const intermediateCommon = ['Thank You', 'Sorry', 'More', 'Water', 'Eat', 'Home'];
    for (const sign of COMMON_SIGNS) {
        map[sign] = intermediateCommon.includes(sign) ? 'Intermediate' : 'Beginner';
    }
    return map;
})();

/**
 * Brief hand-configuration description for each gesture.
 */
export const GESTURE_DESCRIPTIONS: Record<string, string> = {
    // Alphabet — finger extension bitmask descriptions
    // Bitmask order: [Thumb, Index, Middle, Ring, Pinky]
    // 1 = extended, 0 = curled
    'A': 'Fist with thumb alongside fingers — Bitmask: 00000',
    'B': 'Flat hand, fingers together, thumb tucked — Bitmask: 01111',
    'C': 'Curved hand forming a C shape — Bitmask: 11111 (curved)',
    'D': 'Index finger up, others curled to thumb — Bitmask: 01000',
    'E': 'All fingers curled, thumb tucked under — Bitmask: 00000 (curled tips down)',
    'F': 'Thumb and index touching, other fingers up — Bitmask: 00111',
    'G': 'Index and thumb pointing sideways — Bitmask: 11000 (horizontal)',
    'H': 'Index and middle pointing sideways — Bitmask: 01100 (horizontal)',
    'I': 'Pinky extended, others curled — Bitmask: 00001',
    'J': 'Pinky extended, trace J motion — Bitmask: 00001 + motion',
    'K': 'Index and middle up in V, thumb between — Bitmask: 11100',
    'L': 'L-shape: thumb and index extended — Bitmask: 11000',
    'M': 'Three fingers over thumb — Bitmask: 00000 (thumb under 3 fingers)',
    'N': 'Two fingers over thumb — Bitmask: 00000 (thumb under 2 fingers)',
    'O': 'All fingertips touching, forming O — Bitmask: 11111 (rounded)',
    'P': 'Like K but angled downward — Bitmask: 11100 (tilted)',
    'Q': 'Like G but pointing down — Bitmask: 11000 (pointing down)',
    'R': 'Index and middle crossed — Bitmask: 01100 (crossed)',
    'S': 'Fist with thumb over fingers — Bitmask: 00000 (thumb over)',
    'T': 'Fist with thumb between index and middle — Bitmask: 00000 (thumb between)',
    'U': 'Index and middle together, pointing up — Bitmask: 01100',
    'V': 'Index and middle apart in V shape — Bitmask: 01100 (spread)',
    'W': 'Index, middle, ring extended and spread — Bitmask: 01110',
    'X': 'Index finger hooked — Bitmask: 01000 (hooked)',
    'Y': 'Thumb and pinky extended — Bitmask: 10001',
    'Z': 'Index finger traces Z in air — Bitmask: 01000 + motion',
    // Common signs
    'Hello': 'Open hand, all fingers extended — wave motion',
    'Thank You': 'Flat hand from chin outward — forward motion',
    'Yes': 'Closed fist, nodding wrist motion',
    'No': 'Index and middle finger snap to thumb',
    'Please': 'Open palm circles on chest',
    'Sorry': 'Fist circles on chest',
    'Help': 'Fist on open palm, lift upward',
    'More': 'Both hands: fingertips together, tap twice',
    'Stop': 'Flat hand chops onto open palm',
    'Good': 'Flat hand from chin, palm up — outward',
    'Bad': 'Flat hand from chin, palm down — outward',
    'Water': 'W-hand taps chin twice',
    'Eat': 'Flat O hand taps mouth',
    'Home': 'Flat O from cheek to ear',
    'Love': 'Thumb, index, and pinky extended — ILY sign',
};
