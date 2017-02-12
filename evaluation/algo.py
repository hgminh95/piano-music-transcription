# -*- coding: utf-8 -*-


def incif(a, b):
    if a < b:
        return a + 1
    else:
        return a


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def leftjoin(ans, expect, window=0.05):
    ans_len = len(ans)
    expect_len = len(expect)

    i, j = 0, 0
    notes = []
    while i < ans_len and j < expect_len:
        if abs(ans[i][0] - expect[j][0]) <= window:
            notes.append(expect[j][2])
            j = incif(j, expect_len)
        elif ans[i] > expect[j]:
            j = incif(j, expect_len)
        else:
            yield ans[i][0], notes
            i = incif(i, ans_len)
            notes = []
