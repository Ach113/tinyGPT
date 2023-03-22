from data import Dataset
from train import train_bigram, generate_text


def main():
    # define dataset
    dataset = Dataset('tiny_shakespeare.txt')
    train, test = dataset.train_test_split(test_size=.1)
    # train model
    model = train_bigram(train, dataset.vocab_size, batch_size=32, epochs=5000)
    # generate text
    print(generate_text(model, 500, dataset.decode))


if __name__ == '__main__':
    main()
