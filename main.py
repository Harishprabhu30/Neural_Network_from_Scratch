from src.data_loader import load_data_wrapper
from src.basic_network import Network

def main():

    ## load data
    training_data, validation_data, testing_data = load_data_wrapper()

    ## Initializing the Network
    net = Network([784, 30, 10])

    ## Train the Network
    net.SGD(training_data, epochs= 30, mini_batch_size= 10, eta= 3.0, test_data= testing_data)


if __name__ == "__main__":
    main()