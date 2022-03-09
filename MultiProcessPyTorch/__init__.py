

from multiprocessing import Process


def test_loop():
    print("this is a loop")
    i = 0
    while True:
        i += 1
        print("this is a loop ", i)


def test_multiprocessing():
    test_process = Process(target=test_loop)
    test_process.start()


if __name__ == "__main__":
    test_multiprocessing()
