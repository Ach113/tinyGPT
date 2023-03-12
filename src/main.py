from data import Dataset
from train import train_bigram, generate_text


# define dataset
dataset = Dataset('tiny_shakespeare.txt')
data = dataset.data
n = int(.9 * len(data))
train, test = data[:n], data[n:]
# train model
model = train_bigram(train, dataset.vocab_size, batch_size=32, block_size=8, epochs=10000)
# generate text
print(generate_text(model, 300, dataset.decode))
