def to_3_tuple(x):
    return x if type(x) == tuple else (x, x, x )



if __name__ == "__main__":
    x = (2, 2, 2)
    print(type(x) == tuple)
    print(to_3_tuple(x))