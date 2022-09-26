#!/usr/bin/env python
# coding: utf-8

# # (A) Count the number of alphabets in the given string 

# # 1.using isalpha() + len()

# In[1]:


test_str= "welcome to python program "
print("original string is : " + str(test_str))
res = len([l for l in test_str if l.isalpha()])
print(" Count of Alphabets : " + str(res))


# # 2. Using ascii_uppercase() + ascii_lowercase() + len()

# In[5]:


import string
test_str= "My name is Riya Saproo, born on 23/03/02"
res = len([l for l in test_str if l in string.ascii_uppercase or l in string.ascii_lowercase])
print("Count of Alphabets :" + str(res))


# # (B) To extract characters in the given, range from the given string.

# In[6]:


test_list= ["Riya" , "Saproo" , "is" , "my" , "name"]
print("The original list is : " + str(test_list))
strt, end = 10,17
res = ''.join([s for s in test_list])[strt : end]
# printing result
print("Range characters : " + str(res))


# # (c) Check if the string is alphanumeric or not

# # 1.Python String isalnum() Method

# In[7]:


string = "abc 123"
print(string, "is alphanumeric?", string.isalnum())


# # 2. isalnum() in if...else Statement

# In[9]:


A = "RiyaSaproo20"
if A.isalnum():
    print("A is alphanumeric.")
else:
    print("A is not alphanumeric.")


# In[ ]:




