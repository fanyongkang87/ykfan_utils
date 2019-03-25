# class TypedField:
#     def __init__(self, _type):
#         self._type = _type
#
#     def __get__(self, instance, cls):
#         if instance is None:
#             return self
#         else:
#             return getattr(instance, self.name)
#
#     def __set_name__(self, cls, name):
#         self.name = name
#
#     def __set__(self, instance, value):
#         print '========'
#         if not isinstance(value, self._type):
#             raise TypeError('Expected' + str(self._type))
#         instance.__dict__[self.name] = value
#
# class Person:
#     age = TypedField(int)
#     name = TypedField(str)
#
#     def __init__(self, age, name):
#         self.age = age
#         self.name = name
#
# jack = Person(15, 'Jack')
# print jack.name
# jack.age = Person(13, 'lucy')
# print jack.age

# # test int float convert
# # 0.666666666667
# # 0
# # 0.666666666667
#
# print 2.0/3
# print 2/3
# print 2/3.0


# import numpy as np
# print np.random.beta(1.5, 1.5)
#
# label1 = np.array([[1.0, 2.0, 1.0, 2.9], [2.3, 3.2, 2.2, 4.4]])
#
# y1 = np.hstack((label1, np.full((label1.shape[0], 1), np.random.beta(1.5, 1.5))))
# print y1

# from tqdm import tqdm
# import time
# for i in tqdm(range(1000)):
#     print('hello')
#     time.sleep(1)

# from __future__ import print_function
# print('love', 'panda', sep=', ')

# from __future__ import print_function
# try:
#     raise Exception('test')
# except Exception as e_val:
#     print('catch: {}'.format(e_val))
#
# print(2//3)

import os

for root, dirs, files in os.walk('/home/ykfan/code/head_pose/my_utils/test', topdown=False):
    print('==============')
    print(root)
    print(dirs)
    print(files)
