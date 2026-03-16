from activation import LogisticActivation, ReluActivation

act = LogisticActivation()
print(act.activate(5))

act = ReluActivation()
print(act.activate(5))
