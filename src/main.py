from data import Dataset
from train import train_transformer, generate_text


def main():
    # define dataset
    dataset = Dataset('tiny_shakespeare.txt')
    train, test = dataset.train_test_split(test_size=.1)
    # train model
    model = train_transformer(train, dataset.vocab_size)
    # generate text
    print(generate_text(model, 500, dataset.decode))


if __name__ == '__main__':
    main()
