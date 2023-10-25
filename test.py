
def int_point(circle_len):
    res = 0
    circle_x,circle_y  = 0,0
    for circle_x in range(0,circle_len+1):
        for circle_y in range(0,circle_len):
            while circle_x**2 +circle_y**2 == circle_len**2:
                res+=1
                break

    res = 4*(res)
    print(res)

int_point(5)