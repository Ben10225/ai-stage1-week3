from typing import Tuple, List, TypedDict


class FromNode(TypedDict):
    index: int
    ratio: float


class Node(TypedDict):
    from_node: List[FromNode]
    bias: float


class Network:
    def __init__(
        self,
        *,
        layers: List[List[Node]],
    ):
        self.inputs: Tuple[float, ...] = tuple()
        self.layers = layers

    def setInputs(self, *, inputs: Tuple[float, float]) -> None:
        self.inputs = inputs

    def forward(self) -> List[float]:
        results = list(self.inputs)

        for layer in self.layers:
            tmpResult: List[float] = [
                sum(
                    results[from_node["index"]] * from_node["ratio"]
                    for from_node in node["from_node"]
                )
                + node["bias"]
                for node in layer
            ]
            results = tmpResult

        return results


def main():
    # model1
    nn1 = Network(
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

    # model2
    nn2 = Network(
        layers=[
            [
                {
                    "from_node": [
                        {"index": 0, "ratio": 0.5},
                        {"index": 1, "ratio": 1.5},
                    ],
                    "bias": 0.3,
                },
                {
                    "from_node": [
                        {"index": 0, "ratio": 0.6},
                        {"index": 1, "ratio": -0.8},
                    ],
                    "bias": 1.25,
                },
            ],
            [
                {
                    "from_node": [
                        {"index": 0, "ratio": 0.6},
                        {"index": 1, "ratio": -0.8},
                    ],
                    "bias": 0.3,
                },
            ],
            [
                {
                    "from_node": [
                        {"index": 0, "ratio": 0.5},
                    ],
                    "bias": 0.2,
                },
                {
                    "from_node": [
                        {"index": 0, "ratio": -0.4},
                    ],
                    "bias": 0.5,
                },
            ],
        ]
    )

    # task1
    print("------ model1 ------")
    nn1.setInputs(inputs=(1.5, 0.5))
    outputs1 = nn1.forward()
    print(outputs1)

    nn1.setInputs(inputs=(0, 1))
    outputs2 = nn1.forward()
    print(outputs2)

    # task2
    print("------ model2 ------")
    nn2.setInputs(inputs=(0.75, 1.25))
    outputs3 = nn2.forward()
    print(outputs3)

    nn2.setInputs(inputs=(-1, 0.5))
    outputs4 = nn2.forward()
    print(outputs4)


if __name__ == "__main__":
    main()
