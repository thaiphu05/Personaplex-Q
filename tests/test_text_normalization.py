import unicodedata
import unittest

from personaplex_finetuning.text_normalization import (
    decode_vietnamese_telex,
    encode_vietnamese_telex,
    normalize_vietnamese_text,
    strip_vietnamese_diacritics,
)

CORPUS = (
    "Tôi muốn chuyển khoản.",
    "Được rồi, cảm ơn bạn.",
    "Trường hợp này xử lý thế nào?",
    "ă â ê ô ơ ư đ",
    "á à ả ã ạ",
    "ấ ầ ẩ ẫ ậ",
    "ớ ờ ở ỡ ợ",
)

# Words with regular tone placement; ư+ơ lexical exceptions ("được", "mượn",
# "tượng") are excluded from the telex round-trip until a G2P lexicon is used.
ROUND_TRIP_CORPUS = (
    "xin chào",
    "Tôi muốn chuyển khoản",
    "Trường hợp này xử lý thế nào",
)


class DiacriticsStripTest(unittest.TestCase):
    def test_strips_marks_and_stroked_d(self) -> None:
        self.assertEqual(strip_vietnamese_diacritics("xin chào"), "xin chao")
        self.assertEqual(strip_vietnamese_diacritics("Đường"), "Duong")
        self.assertEqual(strip_vietnamese_diacritics("trường hợp"), "truong hop")

    def test_output_is_ascii_for_corpus(self) -> None:
        for text in CORPUS:
            stripped = strip_vietnamese_diacritics(text)
            self.assertTrue(stripped.isascii(), f"non-ascii residue in {stripped!r}")


class TelexTest(unittest.TestCase):
    def test_encode_tone_letters(self) -> None:
        self.assertEqual(encode_vietnamese_telex("chào"), "chaof")
        self.assertEqual(encode_vietnamese_telex("má"), "mas")
        self.assertEqual(encode_vietnamese_telex("hỏi"), "hoir")

    def test_telex_round_trip_preserves_diacritics(self) -> None:
        for text in ROUND_TRIP_CORPUS:
            encoded = encode_vietnamese_telex(text)
            decoded = decode_vietnamese_telex(encoded)
            self.assertEqual(
                unicodedata.normalize("NFC", decoded),
                unicodedata.normalize("NFC", text),
                f"telex round-trip changed {text!r} via {encoded!r} -> {decoded!r}",
            )


class ModeSelectionTest(unittest.TestCase):
    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_vietnamese_text("chào", mode="bogus")

    def test_diacritics_mode_is_identity(self) -> None:
        self.assertEqual(normalize_vietnamese_text("chào", mode="diacritics"), "chào")


class AsciiTokenizer:
    """Stand-in for a tokenizer whose vocabulary covers ASCII only; encoding
    anything non-ascii would have to fall back to an unknown id."""

    unk_id = 0

    def encode(self, text: str) -> list[int]:
        if not text.isascii():
            raise ValueError(f"unk fallback for {text!r}")
        return [ord(character) for character in text]

    def decode(self, ids) -> str:
        return "".join(chr(value) for value in ids)


class TokenizerRoundTripTest(unittest.TestCase):
    def test_normalized_text_encodes_without_unknown_tokens(self) -> None:
        """Raw diacritic text raises (unk); normalized ASCII text encodes."""
        tokenizer = AsciiTokenizer()
        for text in CORPUS:
            with self.assertRaises(ValueError):
                tokenizer.encode(text)
            for mode in ("no_diacritics", "telex"):
                normalized = normalize_vietnamese_text(text, mode=mode)
                ids = tokenizer.encode(normalized)  # must not raise
                self.assertEqual(tokenizer.decode(ids), normalized)


if __name__ == "__main__":
    unittest.main()
