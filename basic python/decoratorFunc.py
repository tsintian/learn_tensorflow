#%%
def messageWithWelcome(str):
    
    def addWelcome():
        return "welcome to "
    
    return addWelcome() + str

# To get site name to which welcome is added.
def site(site_name):
    return site_name

print(messageWithWelcome(site("GeeksforGeeks")))

#%%
# A decorator is a function that takes a function as its only parameter 
# and returns a function. This is helpful to wrap functionality with
# the same code over and over again.

#We use @func_name to specify a decorator to be applied on another function

def decorate_message(fun):
    
    # Nested function
    def addWelcome(site_name2):
        return 'Welcome to ' + fun(site_name2)
    
    # Decorator returns a function
    return addWelcome

@decorate_message
def site2(site_name):
    return site_name

# This call is equivalent to call to
# decorate_message() with function site2("GeeksforGeeks") as parameter
print(site2("GeeksforGeeks"))


#%% 

# Decorator can also be useful to attach data(or add attribute) to functions

def attach_data(func):
    func.data = 3
    return func 

@attach_data
def add(x, y):
    return x + y  

print(add(2, 3))
print(add.data)