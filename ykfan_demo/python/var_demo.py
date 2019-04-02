
# # val scope module/class/fun
# val = 10
#
#
# def fun(use_msg=False):
#     if use_msg:
#         msg = 'var scope!'
#     print(val) #
#     print(msg) # error, no msg assign.
#
#
# fun()


# # [[], [], [], [], []]
# # [[10], [10], [10], [10], [10]]
# # [[10, 20], [10, 20], [10, 20], [10, 20], [10, 20]]
# # [[10, 20], [10, 20], [10, 20], [10, 20], [10, 20], 30]

# a = [[]]*5
# print(a)
# a[0].append(10)
# print(a)
# a[1].append(20)
# print(a)
# a.append(30)
# print(a)

