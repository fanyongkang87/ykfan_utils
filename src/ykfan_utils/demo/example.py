import os.path


def ex_add():
    return 1+2


def package_data_show():
    data_path = os.path.join(os.path.dirname(__file__), 'data.txt')
    file = open(data_path).readlines()
    for line in file:
        print(line)
