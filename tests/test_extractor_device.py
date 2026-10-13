import unittest

from signturk_runtime.extractor import choose_device

CPU_ONLY = ["CPUExecutionProvider"]
WITH_CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class ChooseDeviceTests(unittest.TestCase):
    def test_auto_uses_cuda_when_available(self):
        self.assertEqual(choose_device("auto", WITH_CUDA), "cuda")
        self.assertEqual(choose_device("auto", CPU_ONLY), "cpu")
        self.assertEqual(choose_device("", WITH_CUDA), "cuda")
        self.assertEqual(choose_device(None, WITH_CUDA), "cuda")

    def test_cpu_setting_wins(self):
        self.assertEqual(choose_device(" CPU ", WITH_CUDA), "cpu")

    def test_cuda_setting_needs_the_provider(self):
        self.assertEqual(choose_device("cuda", WITH_CUDA), "cuda")
        self.assertEqual(choose_device("cuda", CPU_ONLY), "cpu")


if __name__ == "__main__":
    unittest.main()
