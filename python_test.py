
#高级api
def check1(str1,str2):
    if(str1 in str2):
        return True
    return False


if __name__=="__main__":
    print(check1("abc","abcd"))