class Perceptron:
    def __init__(self, eta=0.1):
        self.w = None
        self.b = 0.0
        self.eta = eta

    def predict(self, x):
        net_input = sum(w * x_i for w, x_i in zip(self.w, x)) + self.b
        return 1 if net_input > 0 else 0

    def fit(self, X, Y, max_iter=1000):
        self.w = [0.0] * len(X[0])
        self.b = 0.0
        
        for _ in range(max_iter):
            errors = 0
            for x_i, y in zip(X, Y):
                y_hat = self.predict(x_i)
                diff = y - y_hat
                if diff != 0:
                    errors += 1
                    self.w = [w + self.eta * diff * x for w, x in zip(self.w, x_i)]
                    self.b += self.eta * diff
            if errors == 0:
                break

    def print_me(self):
        result = " + ".join(f"{w}*x{index}" for index, w in enumerate(self.w))
        result += f" + {self.b}"
        return result


if __name__ == "__main__":
    perceptron = Perceptron(eta=0.1)
    X = [[140, 0], [10, 1]]  # 140 cm (not red), 10 cm (red)
    Y = [1, 0]  # 1 is banana, 0 is apple
    perceptron.fit(X, Y)
    print(perceptron.predict([100, 0]))
    print(perceptron.print_me())
