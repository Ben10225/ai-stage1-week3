import unittest
from index import Network


class TestNetwork(unittest.TestCase):
    def test_forward(self):
        nn = Network(
            layers=[
                [
                    {
                        "from_node": [
                            {"index": 0, "ratio": 0.5},
                            {"index": 1, "ratio": 0.2},
                        ],
                        "bias": 0.3,
                    },
                    {
                        "from_node": [
                            {"index": 0, "ratio": 0.6},
                            {"index": 1, "ratio": -0.6},
                        ],
                        "bias": 0.25,
                    },
                ],
                [
                    {
                        "from_node": [
                            {"index": 0, "ratio": 0.8},
                            {"index": 1, "ratio": 0.4},
                        ],
                        "bias": -0.5,
                    },
                ],
            ]
        )
        nn.setInputs(inputs=(1, 2))

        res = nn.forward()
        self.assertAlmostEqual(res[0], 0.32, places=5)


if __name__ == "__main__":
    unittest.main()
