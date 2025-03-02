r"""
 _____                 ____  _                       
|_   _|   _ _ __   ___|  _ \(_)_ __  _ __   ___ _ __ 
  | || | | | '_ \ / _ \ |_) | | '_ \| '_ \ / _ \ '__|
  | || |_| | | | |  __/  _ <| | |_) | |_) |  __/ |   
  |_| \__,_|_| |_|\___|_| \_\_| .__/| .__/ \___|_|   
                              |_|   |_|              
Description: Convert CSV file to ABC notation and optionally transpose to a target key
Assumes the following CSV format:
setting_id,name,type,meter,mode,date,username,abc

Usage: python csv_to_abc_transposer.py [-h] [-o OUTPUT] [-k KEY] csv_file
Example: python .\csv_to_abc_transposer.py .\raw_data\tunes.csv -o output_dmajor.abc -k Dmajor

This script borrows heavily from the following JavaScript implementation: https://github.com/paulrosen/abcjs/blob/main/src/str/output.js
Clause 3.7 Sonnet did much of the heavy lifiting in the conversion of the JavaScript to Python
"""

import csv
import re
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Set

# Key signature mapping for transposition
KEY_SIGNATURES = {
    "Cmajor": {"root": "C", "acc": "", "mode": "major"},
    "Gmajor": {"root": "G", "acc": "", "mode": "major"},
    "Dmajor": {"root": "D", "acc": "", "mode": "major"},
    "Amajor": {"root": "A", "acc": "", "mode": "major"},
    "Emajor": {"root": "E", "acc": "", "mode": "major"},
    "Bmajor": {"root": "B", "acc": "", "mode": "major"},
    "F#major": {"root": "F", "acc": "#", "mode": "major"},
    "C#major": {"root": "C", "acc": "#", "mode": "major"},
    "Fmajor": {"root": "F", "acc": "", "mode": "major"},
    "Bbmajor": {"root": "B", "acc": "b", "mode": "major"},
    "Ebmajor": {"root": "E", "acc": "b", "mode": "major"},
    "Abmajor": {"root": "A", "acc": "b", "mode": "major"},
    "Dbmajor": {"root": "D", "acc": "b", "mode": "major"},
    "Gbmajor": {"root": "G", "acc": "b", "mode": "major"},
    "Aminor": {"root": "A", "acc": "", "mode": "minor"},
    "Eminor": {"root": "E", "acc": "", "mode": "minor"},
    "Bminor": {"root": "B", "acc": "", "mode": "minor"},
    "F#minor": {"root": "F", "acc": "#", "mode": "minor"},
    "C#minor": {"root": "C", "acc": "#", "mode": "minor"},
    "G#minor": {"root": "G", "acc": "#", "mode": "minor"},
    "D#minor": {"root": "D", "acc": "#", "mode": "minor"},
    "Dminor": {"root": "D", "acc": "", "mode": "minor"},
    "Gminor": {"root": "G", "acc": "", "mode": "minor"},
    "Cminor": {"root": "C", "acc": "", "mode": "minor"},
    "Fminor": {"root": "F", "acc": "", "mode": "minor"},
    "Bbminor": {"root": "B", "acc": "b", "mode": "minor"},
    "Ebminor": {"root": "E", "acc": "b", "mode": "minor"},
    "Dorian": {"root": "D", "acc": "", "mode": "dorian"},
    "Mixolydian": {"root": "G", "acc": "", "mode": "mixolydian"},
}

# Notes and their semitone values
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_VALUES = {note: i for i, note in enumerate(NOTES)}
NOTE_VALUES.update(
    {
        "Db": NOTE_VALUES["C#"],
        "Eb": NOTE_VALUES["D#"],
        "Gb": NOTE_VALUES["F#"],
        "Ab": NOTE_VALUES["G#"],
        "Bb": NOTE_VALUES["A#"],
    }
)

# Letters used in music notation
LETTERS = "CDEFGAB"

# Map for accidentals
ACCIDENTALS = {"b": -1, "#": 1, "": 0}  # flat  # sharp  # natural

# Regular expressions for parsing ABC notation
NOTE_REGEX = re.compile(r"([_^=]*)([A-Ga-g])([,\']*)")


def convert_to_abc(
    csv_file: str, output_file: Optional[str] = None, target_key: Optional[str] = None
) -> str:
    """
    Convert CSV file to ABC notation and optionally transpose to a target key

    Args:
        csv_file: Path to the CSV file
        output_file: Path to the output file (optional)
        target_key: Key to transpose to (optional)

    Returns:
        The generated ABC notation as a string
    """
    tunes = []
    original_keys = {}

    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Store original key for possible transposition
            original_keys[row["setting_id"]] = row["mode"]

            # Format ABC notation
            abc_notation = f"""X:{row['setting_id']}
T:{row['name']}
R:{row['type']}
M:{row['meter']}
L:1/8
K:{row['mode']}
% date: {row['date']}
% user: {row['username']}
{row['abc']}"""

            tunes.append(abc_notation)

    # Join all tunes with double newlines
    result = "\n\n".join(tunes)

    # Transpose if requested
    if target_key:
        result = transpose_abc(result, original_keys, target_key)

    # Write to output file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(result)

    return result


def get_semitone_distance(source_key: str, target_key: str) -> int:
    """
    Calculate semitone distance between two keys

    Args:
        source_key: Original key
        target_key: Target key

    Returns:
        Number of semitones to transpose (positive or negative)
    """
    if not source_key or not target_key:
        return 0

    source = KEY_SIGNATURES.get(source_key)
    target = KEY_SIGNATURES.get(target_key)

    if not source or not target:
        return 0

    source_note = source["root"] + source["acc"]
    target_note = target["root"] + target["acc"]

    source_value = NOTE_VALUES.get(source_note, 0)
    target_value = NOTE_VALUES.get(target_note, 0)

    semitones = target_value - source_value

    # Adjust for octave to find shortest distance
    if semitones < -6:
        semitones += 12
    if semitones > 6:
        semitones -= 12

    return semitones


def get_key_signature_notes(key: str) -> Dict[str, str]:
    """
    Get dictionary of notes and their accidentals in a key signature

    Args:
        key: Key signature (e.g., 'Gmajor')

    Returns:
        Dictionary mapping note letters to accidentals
    """
    # Key signature accidentals for major keys
    key_accidentals = {
        "Cmajor": {},
        "Gmajor": {"F": "#"},
        "Dmajor": {"F": "#", "C": "#"},
        "Amajor": {"F": "#", "C": "#", "G": "#"},
        "Emajor": {"F": "#", "C": "#", "G": "#", "D": "#"},
        "Bmajor": {"F": "#", "C": "#", "G": "#", "D": "#", "A": "#"},
        "F#major": {"F": "#", "C": "#", "G": "#", "D": "#", "A": "#", "E": "#"},
        "C#major": {
            "F": "#",
            "C": "#",
            "G": "#",
            "D": "#",
            "A": "#",
            "E": "#",
            "B": "#",
        },
        "Fmajor": {"B": "b"},
        "Bbmajor": {"B": "b", "E": "b"},
        "Ebmajor": {"B": "b", "E": "b", "A": "b"},
        "Abmajor": {"B": "b", "E": "b", "A": "b", "D": "b"},
        "Dbmajor": {"B": "b", "E": "b", "A": "b", "D": "b", "G": "b"},
        "Gbmajor": {"B": "b", "E": "b", "A": "b", "D": "b", "G": "b", "C": "b"},
    }

    # Add minor keys (relative minors)
    key_accidentals.update(
        {
            "Aminor": {},
            "Eminor": {"F": "#"},
            "Bminor": {"F": "#", "C": "#"},
            "F#minor": {"F": "#", "C": "#", "G": "#"},
            "C#minor": {"F": "#", "C": "#", "G": "#", "D": "#"},
            "G#minor": {"F": "#", "C": "#", "G": "#", "D": "#", "A": "#"},
            "D#minor": {"F": "#", "C": "#", "G": "#", "D": "#", "A": "#", "E": "#"},
            "Dminor": {"B": "b"},
            "Gminor": {"B": "b", "E": "b"},
            "Cminor": {"B": "b", "E": "b", "A": "b"},
            "Fminor": {"B": "b", "E": "b", "A": "b", "D": "b"},
            "Bbminor": {"B": "b", "E": "b", "A": "b", "D": "b", "G": "b"},
            "Ebminor": {"B": "b", "E": "b", "A": "b", "D": "b", "G": "b", "C": "b"},
        }
    )

    return key_accidentals.get(key, {})


def transpose_note(
    note_match: re.Match,
    semitones: int,
    source_key_accidentals: Dict[str, str],
    target_key_accidentals: Dict[str, str],
    measure_accidentals: Dict[str, str],
) -> str:
    """
    Transpose a single note by the given number of semitones

    Args:
        note_match: Regex match object for the note
        semitones: Number of semitones to transpose
        source_key_accidentals: Accidentals in the source key
        target_key_accidentals: Accidentals in the target key
        measure_accidentals: Accidentals already present in the current measure

    Returns:
        Transposed note
    """
    if not note_match:
        return ""

    # Extract note components
    accidental = note_match.group(1) or ""
    note_letter = note_match.group(2)
    octave_marker = note_match.group(3) or ""

    # Determine if note is lowercase (higher octave)
    is_lowercase = note_letter.islower()
    note_letter_upper = note_letter.upper()

    # Calculate current note index
    letter_index = LETTERS.index(note_letter_upper)

    # Calculate semitone position in the scale
    current_semitone = NOTE_VALUES[note_letter_upper]

    # Adjust for accidentals
    if note_letter_upper in source_key_accidentals:
        current_semitone += ACCIDENTALS.get(
            source_key_accidentals[note_letter_upper], 0
        )

    # Adjust for explicit accidentals in the notation
    if accidental:
        if accidental == "^":
            current_semitone += 1
        elif accidental == "_":
            current_semitone -= 1
        elif accidental == "=":
            # Natural - reset to the base note
            pass

    # Calculate target semitone
    target_semitone = (current_semitone + semitones) % 12

    # Find the closest letter name in the target key
    letter_steps = semitones // 12 * 7  # Full octave changes
    steps_within_octave = semitones % 12

    # Map of semitone distances between letters (0=unison, 1=minor 2nd, 2=major 2nd, etc.)
    semitone_steps = {
        0: 0,  # C->C
        1: 1,  # C->C# (or C->Db)
        2: 1,  # C->D
        3: 2,  # C->D# (or C->Eb)
        4: 2,  # C->E
        5: 3,  # C->F
        6: 3,  # C->F# (or C->Gb)
        7: 4,  # C->G
        8: 5,  # C->G# (or C->Ab)
        9: 5,  # C->A
        10: 6,  # C->A# (or C->Bb)
        11: 6,  # C->B
    }

    # Calculate letter steps within octave
    letter_steps += semitone_steps.get(steps_within_octave, 0)

    # Calculate new letter index
    new_letter_index = (letter_index + letter_steps) % 7
    new_letter = LETTERS[new_letter_index]

    # Determine if we need to adjust octave
    octave_shift = (letter_index + letter_steps) // 7

    # Calculate new octave markers
    if is_lowercase:
        # Handle lowercase notes (higher octave)
        if octave_marker:
            # Count the number of ' markers
            apostrophe_count = octave_marker.count("'")
            new_octave_marker = "'" * (apostrophe_count + octave_shift)
        else:
            if octave_shift > 0:
                new_octave_marker = "'" * octave_shift
            elif octave_shift < 0:
                # Change to uppercase if shifting down an octave
                is_lowercase = False
                new_octave_marker = ""
            else:
                new_octave_marker = ""
    else:
        # Handle uppercase notes (lower or middle octave)
        if octave_marker:
            # Count the number of , markers
            comma_count = octave_marker.count(",")
            new_comma_count = comma_count - octave_shift

            if new_comma_count < 0:
                # Change to lowercase if shifting up an octave
                is_lowercase = True
                new_octave_marker = "'" * abs(new_comma_count - 1)
            else:
                new_octave_marker = "," * new_comma_count
        else:
            if octave_shift > 0:
                # Change to lowercase if shifting up an octave
                is_lowercase = True
                new_octave_marker = "'" * (octave_shift - 1)
            elif octave_shift < 0:
                new_octave_marker = "," * abs(octave_shift)
            else:
                new_octave_marker = ""

    # Calculate the natural semitone of the new note letter
    new_note_semitone = NOTE_VALUES[new_letter]

    # Calculate the required accidental
    target_accidental_in_key = target_key_accidentals.get(new_letter, "")

    # Adjust for key signature
    if target_accidental_in_key:
        new_note_semitone += ACCIDENTALS.get(target_accidental_in_key, 0)

    # Determine if we need an explicit accidental
    explicit_accidental = ""
    if new_note_semitone != target_semitone:
        semitone_diff = target_semitone - new_note_semitone

        if semitone_diff == 1:
            explicit_accidental = "^"  # Sharp
        elif semitone_diff == -1:
            explicit_accidental = "_"  # Flat
        elif semitone_diff == 2:
            explicit_accidental = "^^"  # Double sharp
        elif semitone_diff == -2:
            explicit_accidental = "__"  # Double flat
        elif semitone_diff == 0:
            # Check if we need a natural sign to override a key signature
            if target_accidental_in_key and new_letter not in measure_accidentals:
                explicit_accidental = "="  # Natural

    # Format the new note
    new_note = (
        explicit_accidental
        + (new_letter.lower() if is_lowercase else new_letter)
        + new_octave_marker
    )

    # Update measure accidentals
    if explicit_accidental:
        measure_accidentals[new_letter] = explicit_accidental

    return new_note


def transpose_abc_line(
    line: str, semitones: int, source_key: str, target_key: str
) -> str:
    """
    Transpose a line of ABC notation

    Args:
        line: Line of ABC notation
        semitones: Number of semitones to transpose
        source_key: Original key
        target_key: Target key

    Returns:
        Transposed line
    """
    # Skip metadata and comments
    if (
        line.startswith("%")
        or line.startswith("X:")
        or line.startswith("T:")
        or line.startswith("R:")
        or line.startswith("M:")
        or line.startswith("L:")
        or line.startswith("K:")
    ):
        if line.startswith("K:"):
            return f"K:{target_key}"
        return line

    # Get key signature accidentals
    source_key_accidentals = get_key_signature_notes(source_key)
    target_key_accidentals = get_key_signature_notes(target_key)

    # Keep track of measure accidentals
    measure_accidentals = {}

    # Process the line character by character to handle complex ABC syntax
    result = ""
    i = 0
    while i < len(line):
        # Check for measure bars that reset accidentals
        if line[i] == "|":
            result += line[i]
            measure_accidentals = {}
            i += 1
            continue

        # Check for notes
        note_match = NOTE_REGEX.match(line[i:])
        if note_match:
            transposed_note = transpose_note(
                note_match,
                semitones,
                source_key_accidentals,
                target_key_accidentals,
                measure_accidentals,
            )
            result += transposed_note
            i += note_match.end()
        else:
            # Copy non-note characters
            result += line[i]
            i += 1

    return result


def transpose_abc(abc: str, original_keys: Dict[str, str], target_key: str) -> str:
    """
    Transpose ABC notation to a target key

    Args:
        abc: Original ABC notation
        original_keys: Dictionary mapping setting_id to original key
        target_key: Target key

    Returns:
        Transposed ABC notation
    """
    # Split ABC by tunes
    tunes = re.split(r"\n\n(?=X:)", abc)
    transposed_tunes = []

    for tune in tunes:
        # Extract setting ID to find original key
        setting_id_match = re.search(r"X:(\d+)", tune)
        if not setting_id_match:
            transposed_tunes.append(tune)
            continue

        setting_id = setting_id_match.group(1)
        original_key = original_keys.get(setting_id)

        if not original_key or not target_key:
            transposed_tunes.append(tune)
            continue

        # Calculate semitone distance
        semitones = get_semitone_distance(original_key, target_key)

        # Process each line
        lines = tune.split("\n")
        transposed_lines = [
            transpose_abc_line(line, semitones, original_key, target_key)
            for line in lines
        ]

        transposed_tunes.append("\n".join(transposed_lines))

    return "\n\n".join(transposed_tunes)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV to ABC notation with transposition"
    )
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("-o", "--output", help="Output ABC file")
    parser.add_argument("-k", "--key", help="Target key for transposition")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        if args.verbose:
            print(f"Converting {args.csv_file} to ABC notation...", file=sys.stderr)
            if args.key:
                print(f"Transposing to {args.key}...", file=sys.stderr)

        result = convert_to_abc(args.csv_file, args.output, args.key)

        # If no output file is specified, print to stdout
        if not args.output:
            print(result)

        if args.verbose:
            print(
                f"Conversion {'and transposition ' if args.key else ''}completed successfully!",
                file=sys.stderr,
            )
            if args.output:
                print(f"Output written to {args.output}", file=sys.stderr)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
