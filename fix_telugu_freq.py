import re

def convert_telugu_digits_to_ascii(text):
    mapping = {
        ord('౦'): '0', ord('౧'): '1', ord('౨'): '2', ord('౩'): '3',
        ord('౪'): '4', ord('౫'): '5', ord('౬'): '6', ord('౭'): '7',
        ord('౮'): '8', ord('౯'): '9'
    }
    return text.translate(mapping)

infile = "app/data/te_word_frequency.txt"
outfile = "app/data/te_word_frequency_ascii.txt"

with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, start=1):
        if not line.strip():
            continue
        # split on tab OR whitespace
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 2:
            print(f"Skipping malformed line {i}: {line.strip()}")
            continue
        word = parts[0]
        freq = parts[1]
        freq_ascii = convert_telugu_digits_to_ascii(freq.strip())
        fout.write(f"{word}\t{freq_ascii}\n")

print("Converted file written to", outfile)
