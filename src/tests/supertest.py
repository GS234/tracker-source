class A:
    def __init__(self, x=13):
        self.x  = x
    
    def newstr(self):
        return "new str: this is A"


    def __str__(self):
        # return "this is A"
        return self.newstr()

class B(A):
    def __init__(self, x=12, b=1):
        super().__init__(x)
        self.b=b
    
    def newstr(self):
        return "abcabc: B"

    # def __str__(self):
    #     return "this is B"


def main():
    a = A(100)
    print(a.x)

    b = B(18, 2)
    print(b.x, b.b)
    # works as intended

    print(a, b)



if __name__=="__main__":
    main()