# Train for recognizing which bitstrings represent multiples of some factor
# factor 2 is easy (only need to consider last bit), factor 3 is hard (all bits affect result in some way)

from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sys import argv


def generateTestcases(amount, size, divisor):
  testcases, outputs = [], []
  for i in range(amount):
    shouldBeDivisible = True if np.random.randint(0, 2) == 0 else False
    v = np.random.randint(0, 2**size)
    while (shouldBeDivisible and v % divisor != 0) or (not shouldBeDivisible and v % divisor == 0):
      v = np.random.randint(0, 2**size)
    bitstring = []
    vPart = v
    while vPart > 0:
      bitstring.append(vPart % 2)
      vPart = vPart // 2
    bitstring += [0] * (size - len(bitstring))
    bitstring = list(reversed(bitstring))
    testcases.append(bitstring)
    outputs.append(1 if shouldBeDivisible else 0)
  return (np.array(testcases), np.array(outputs))


if len(argv) != 5:
  print("usage: divisible.py datasetSize bits divisor epochs")
  print("example: divisible.py 1000 16 2 10")
  exit(1)


datasetSize, datasetBits, divisor, epochs = int(argv[1]), int(argv[2]), int(argv[3]), int(argv[4])
inData, outData = generateTestcases(datasetSize, datasetBits, divisor)

model = Sequential()
model.add(Dense(datasetBits, input_dim = datasetBits, activation = "sigmoid"))
model.add(Dense(datasetBits, activation = "sigmoid"))
model.add(Dense(1, activation = "sigmoid"))
model.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(inData, outData, epochs = epochs, batch_size = 10)

inDataTest, outDataTest = generateTestcases(datasetSize, datasetBits, divisor)
print("testcase accuracy: {}".format(model.evaluate(inDataTest, outDataTest)[1]))
