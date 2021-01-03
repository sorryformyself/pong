class a:
    pass

class b:
    def __init__(self, arg):
        self.arg = arg

test1 = a()
test2 = b(test1)

print(test2.arg is test1)