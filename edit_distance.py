# coding:utf-8
def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1
    i=None
    j=None
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

if __name__ == '__main__':
    print(edit_distance(u"蚂蚁 借呗 一共 能 借 多少 钱".split(), u"我 一共 借 了 花呗 多少 钱".split()))