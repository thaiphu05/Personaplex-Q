import unittest

from personaplex_finetuning.qwen_tokenizer import QwenTokenizer


class FakeQwenTokenizer:
    pad_token_id = 151643
    eos_token_id = 151645

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:
            return [self.eos_token_id, len(text), self.eos_token_id]
        return [len(text)]


class QwenTokenizerTest(unittest.TestCase):
    def test_exposes_padding_and_end_padding_ids(self) -> None:
        tokenizer = QwenTokenizer(FakeQwenTokenizer())
        self.assertEqual(tokenizer.padding_id, 151643)
        self.assertEqual(tokenizer.end_padding_id, 151645)

    def test_encodes_without_special_tokens(self) -> None:
        tokenizer = QwenTokenizer(FakeQwenTokenizer())
        self.assertEqual(tokenizer.encode("xin chào"), [8])

    def test_padding_falls_back_to_eos_when_missing(self) -> None:
        class NoPadTokenizer(FakeQwenTokenizer):
            pad_token_id = None

        tokenizer = QwenTokenizer(NoPadTokenizer())
        self.assertEqual(tokenizer.padding_id, 151645)

    def test_end_padding_override(self) -> None:
        tokenizer = QwenTokenizer(FakeQwenTokenizer(), end_padding_id=7)
        self.assertEqual(tokenizer.end_padding_id, 7)


if __name__ == "__main__":
    unittest.main()
