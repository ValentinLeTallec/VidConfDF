from sys import argv


def generateCommands(personA: str, personB: str, it = 200000):
    extract1 = "faceswap extract -i data/src/{0}/ -o data/out/{0}_faces"
    extract2 = "faceswap extract -i data/src/{0}/ -o data/out/{0}_faces"
    train = "faceswap train -A data/out/{0}_faces -B data/out/{1}_faces -m data/out/models/{0}_{1} -it {2} -s 1000"
    convert = "faceswap convert -i data/src/{0}/ -o data/out/{0}_{1} -m data/out/models/{0}_{1}"

    print(extract1.format(personA))
    print(extract2.format(personB))
    print(train.format(personA, personB, it))
    print(convert.format(personA, personB))

if __name__ == '__main__':
    args = argv[1:]
    if len(args) < 2 and len(args) > 3:
        print("Error : 2 or 3 arguments expected")
    elif len(args) == 2:
        generateCommands(args[0], args[1])
    elif len(args) == 3:
        generateCommands(args[0], args[1], args[2])
