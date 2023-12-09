import unittest
from mlx_transformer import flash_attn


class TestExists(unittest.TestCase):
    def test_exists_none(self):
        self.assertEqual(flash_attn.exists(None), False)

    def test_exists_not_none(self):
        self.assertEqual(flash_attn.exists(5), True)
        self.assertEqual(flash_attn.exists("Hello"), True)
        self.assertEqual(flash_attn.exists([]), True)
        self.assertEqual(flash_attn.exists([1, 2, 3]), True)
        self.assertEqual(flash_attn.exists({}), True)
        self.assertEqual(flash_attn.exists({"key": "value"}), True)
        self.assertEqual(flash_attn.exists(set()), True)
        self.assertEqual(flash_attn.exists({1, 2, 3}), True)
        self.assertEqual(flash_attn.exists(0), True)
        self.assertEqual(flash_attn.exists(""), True)
        self.assertEqual(flash_attn.exists(False), True)
        self.assertEqual(flash_attn.exists(()), True)
        self.assertEqual(flash_attn.exists((1, 2, 3)), True)
        self.assertEqual(flash_attn.exists(0.0), True)
        self.assertEqual(flash_attn.exists(float("inf")), True)
        self.assertEqual(flash_attn.exists(float("-inf")), True)
        self.assertEqual(flash_attn.exists(float("nan")), True)
        self.assertEqual(flash_attn.exists(b""), True)
        self.assertEqual(flash_attn.exists(b"bytes"), True)
        self.assertEqual(flash_attn.exists(bytearray()), True)
        self.assertEqual(flash_attn.exists(bytearray(b"bytes")), True)
        self.assertEqual(flash_attn.exists(memoryview(b"bytes")), True)
        self.assertEqual(flash_attn.exists(range(0)), True)
        self.assertEqual(flash_attn.exists(range(10)), True)
        self.assertEqual(flash_attn.exists(slice(0)), True)
        self.assertEqual(flash_attn.exists(slice(10)), True)
        self.assertEqual(flash_attn.exists(Ellipsis), True)
        self.assertEqual(flash_attn.exists(NotImplemented), True)
        self.assertEqual(flash_attn.exists(complex(0)), True)
        self.assertEqual(flash_attn.exists(complex(1, 1)), True)


if __name__ == "__main__":
    unittest.main()
