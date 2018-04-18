
# coding: utf-8

# In[2]:

from urllib.request import urlopen
#statements and exp
shakespeare=urlopen('http://composingprograms.com/shakespeare.txt')
#Functions
words=set(shakespeare.read().decode().split())
#objects
{w for w in words if len(w)==6 and w[::-1] in words}


# In[1]:

#Elements of Programming

42 #Expressions

1/2+1/4+1/5#Compound Expressions with infix notations

max(min(1,-2),min(pow(3,5),-4))#Call Expressions

from operator import add, mul, sub #importing library functions(modules and their named attributes)

sub(100,mul(7,add(8,4)))

#Names and the Environment
f=max
f(2,3,4)
pi=3.14
radius=2
area,circumference=pi*radius*radius,2*pi*radius
area
circumference

#Evaluating Nested Expressions
sub(pow(2,add(1,10)),pow(2,5))

abs(-2) #pure functions
print(1,2,3)#non-pure function-cannot be an expression in an assignment statement
print(print(1), print(2))




# In[4]:

#Control and Testing

#Fibonacci

def fib(n):
    pred,curr=0,1
    k=2
    while k<n:
        pred,curr=curr,pred+curr
        k=k+1
    return curr
result=fib(8)

assert fib(2)==1

def sum_naturals(n):
        """Return the sum of the first n natural numbers.

        >>> sum_naturals(10)
        55
        >>> sum_naturals(100)
        5050
        """
        total, k = 0, 1
        while k <= n:
            total, k = total + k, k + 1
        return total
from doctest import testmod
testmod()
from doctest import run_docstring_examples
run_docstring_examples(sum_naturals, globals(), True)






# In[12]:

#Higher Order Functions

#Functions as arguments

def summation(n,term):
    total, k=0,1
    while k<=n:
        total,k=total+term(k), k+1
    return total

def cube(x):
    return x*x*x

def sum_cubes(n):
    return summation(n,cube)
result=sum_cubes(3)

#Functions as general methods

def improve(update,close,guess=1):
    while not close(guess):
        guess=update(guess)
    return guess

def golden_update(guess):
    return 1/guess + 1

def square_close_to_successor(guess):
    return approx_eq(guess*guess, guess+1)

def approx_eq(x,y,tolerance=1e-3):
    return abs(x-y)<tolerance

phi=improve(golden_update, square_close_to_successor)

#Nested functions

def average(x,y):
    return (x+y)/2
def improve(update,close,guess=1):
    while not close(guess):
        guess=update(guess)
    return guess
def approx_eq(x,y,tolerance=1e-3):
    return abs(x-y)<tolerance
def sqrt(a):
    def sqrt_update(x):
        return average(a,a/x)
    def sqrt_close(x):
        return approx_eq(x*x, a)
    return improve(sqrt_update,sqrt_close)
result=sqrt(256)

#functions as returned values- Newton's method

def newton_update(f, df):
        def update(x):
            return x - f(x) / df(x)
        return update
    
def find_zero(f, df):
    def near_zero(x):
        return approx_eq(f(x), 0)
    return improve(newton_update(f, df), near_zero)

def square_root_newton(a):
    def f(x):
        return x * x - a
    def df(x):
        return 2 * x
    return find_zero(f, df)

def nth_root_of_a(n, a):
        def f(x):
            return power(x, n) - a
        def df(x):
            return n * power(x, n-1)
        return find_zero(f, df)
    
nth_root_of_a(2, 64)

#Currying

def curry_pow(x):
    def h(y):
        return pow(x,y)
    return h
curry_pow(2,3)

#Lambda Expressions

def compose1(f,g):
    return lambda x: f(g(x))

f=compose1(lambda x:x*x, lambda y:y+1)

result=f(10)

#Function Decorators

def trace(fn):
        def wrapped(x):
            print('-> ', fn, '(', x, ')')
            return fn(x)
        return wrapped
@trace
def triple(x):
    return 3 * x







# In[5]:

#Recursion

#mutually recursive
def is_even(n):
    if n==0:
        return True
    else:
        return is_odd(n-1)
    
def is_odd(n):
    if n==0:
        return False
    else:
        return is_even(n-1)
result=is_even(4)

#Tree recursion

def fib(n):
    if n==1:
        return 0
    if n==2:
        return 1
    else:
        return fib(n-1)+fib(n-1)
result=fib(6)


# In[6]:

#Data types
type(2)#built in data types
type(1.5)#float
type(1+1j)#complex

#Data Abstraction

def add_rationals(x, y):
        nx, dx = numer(x), denom(x)
        ny, dy = numer(y), denom(y)
        return rational(nx * dy + ny * dx, dx * dy)
def mul_rationals(x, y):
        return rational(numer(x) * numer(y), denom(x) * denom(y))
def print_rational(x):
        print(numer(x), '/', denom(x))
def rationals_are_equal(x, y):
        return numer(x) * denom(y) == numer(y) * denom(x)
from fractions import gcd
def rational(n,d):
    g=gcd(n,d)
    return[n//g,d//g]
def numer(x):
    return x[0]
def denom(x):
    return x[1]
third = rational(1, 3)
print_rational(add_rationals(third, third))    


# In[7]:

#sequences
digits = [1, 8, 2, 8]
len(digits)
#sequence iteration
def count(s,value):
    total=0
    for elem in s:
        if elem==value:
            total=total+1
    return total
count(digits, 8)

#sequence unpacking
pairs=[[1,2],[2,2],[2,3],[4,4]]
same_count=0
for x, y in pairs:
    if x==y:
        same_count=same_count+1
same_count

for _ in range(3):
    print('Go Bears!')

#sequence processing
#List Compreension
odds = [1, 3, 5, 7, 9]
[x+1 for x in odds]
#Aggregation
def divisors(n):
    return [1]+[x for x in range(2,n) if n%x==0]
[n for n in range(1, 1000) if sum(divisors(n)) == n]


#higher order functions
from functools import reduce
from operator import mul
apply_to_all=lambda map_fn, s: list(map(map_fn,s))
keep_if=lambda filter_fn, s:list(filter(filter_fn,s))
def product(s):
    return reduce(mul,s)

#sequence abstraction
#membership
2 in digits
#slicing
digits[0:2]
#strings
'I am a string!'
city='Berkeley'
len(city)
#Trees
def tree(root, branches=[]):
        for branch in branches:
            assert is_tree(branch), 'branches must be trees'
        return [root] + list(branches)
def root(tree):
        return tree[0]
def branches(tree):
        return tree[1:]
def is_tree(tree):
        if type(tree) != list or len(tree) < 1:
            return False
        for branch in branches(tree):
            if not is_tree(branch):
                return False
        return True
def is_leaf(tree):
        return not branches(tree)
t = tree(3, [tree(1), tree(2, [tree(1), tree(1)])])
t

#Linked Lists
empty = 'empty'
def first(s):
    return s[0]
def rest(s):
    return s[1]

def len_link_recursive(s):
        """Return the length of a linked list s."""
        if s == empty:
            return 0
        return 1 + len_link_recursive(rest(s))
def getitem_link_recursive(s, i):
        """Return the element at index i of linked list s."""
        if i == 0:
            return first(s)
        return getitem_link_recursive(rest(s), i - 1)
four = [1, [2, [3, [4, 'empty']]]]
len_link_recursive(four)


# In[8]:

#Mutable data
from datetime import date
chinese=['coin','string','myriad']
suits=chinese
suits.pop()
suits.remove('string')
suits.append('cup')
suits.extend(['sword','club'])
suits[2]='spade'
suits[0:2]=['heart','diamond']
nest=list(suits)
nest[0]=suits
suits.insert(2,'Joker')
joker=nest[0].pop(2)
joker
suits is nest[0]
#Tuples
(1,2+3,"the")
code = ("up", "up", "down", "down") + ("left", "right") * 2
len(code)
code[3]
code.index("right")

#Dictionaries

numerals = {'I': 1.0, 'V': 5, 'X': 10}
numerals['X']
numerals['L']=50#unordered
sum(numerals.values())
dict([(3,9),(4,16),(5,25)])
{x: x*x for x in range(3,6)}#dictionary comprehension

#Local State
def make_withdraw(balance):
    def withdraw(amount):
        nonlocal balance
        if amount>balance:
            return 'Insuff funds'
        balance=balance-amount
        return balance
    return withdraw
wd=make_withdraw(20)
wd(5)
wd(3)



# In[9]:

#Implementing Lists and Dictionaries
def mutable_link():
        """Return a functional implementation of a mutable linked list."""
        contents = empty
        def dispatch(message, value=None):
            nonlocal contents
            if message == 'len':
                return len_link(contents)
            elif message == 'getitem':
                return getitem_link(contents, value)
            elif message == 'push_first':
                contents = link(value, contents)
            elif message == 'pop_first':
                f = first(contents)
                contents = rest(contents)
                return f
            elif message == 'str':
                return join_link(contents, ", ")
        return dispatch
#dictionaries
def dictionary():
        
        """Return a functional implementation of a dictionary."""
        records = []
        def getitem(key):
            matches = [r for r in records if r[0] == key]
            if len(matches) == 1:
                key, value = matches[0]
                return value
        def setitem(key, value):
            nonlocal records
            non_matches = [r for r in records if r[0] != key]
            records = non_matches + [[key, value]]
        def dispatch(message, key=None, value=None):
            if message == 'getitem':
                return getitem(key)
            elif message == 'setitem':
                setitem(key, value)
        return dispatch
d = dictionary()
d('setitem', 4, 16)
d('getitem', 3)
    


# In[10]:

#Object Oriented Programming
#initialization
class Account:
    def __init__(self, account_holder):
        self.balance=0
        self.holder=account_holder
a=Account('kirk')
#Identity
b=Account('Spock')
b.balance=200
[acc.balance for acc in (a,b)]

#Methods
class Account:
    def __init__(self,account_holder):
        self.balance=0
        self.holder=account_holder
    def deposit(self,amount):
        self.balance=self.balance+amount
        return self.balance
    def withdraw(self,amount):
        if amount>self.balance:
            return 'insufficient funds'
        self.balance=self.balance-amount
        return self.balance
spock_account = Account('Spock')
spock_account.deposit(100)


# In[11]:

#inheritance
class CheckingAccount(Account):
    withdraw_charge=1
    interest=0.01
    def withdraw(self, amount):
        return Account.withdraw(self,amount+self.withdraw_charge)

checking = CheckingAccount('Sam')
checking.deposit(10)

#Interface
def deposit_all(winners, amount=5):
        for account in winners:
            account.deposit(amount)
            
#Multiple Inheritance
class SavingsAccount(Account):
        deposit_charge = 2
        def deposit(self, amount):
            return Account.deposit(self, amount - self.deposit_charge)
class AsSeenOnTVAccount(CheckingAccount, SavingsAccount):
        def __init__(self, account_holder):
            self.holder = account_holder
            self.balance = 1   
such_a_deal = AsSeenOnTVAccount("John")
such_a_deal.deposit(20)



# In[12]:

#Object Abstraction

from datetime import date
tues=date(2011,9,12)
repr(tues)
str(tues)
tues.__repr__()
Account.__bool__ = lambda self: self.balance != 0
bool(Account('Jack'))
if not Account('Jack'):
        print('Jack has nothing')
#sequence operations
'Go Bears!'.__len__()
#Callable Objects
class Adder(object):
    def __init__(self, n):
            self.n = n
    def __call__(self, k):
            return self.n + k
add_three_obj=Adder(3)
add_three_obj(4)



# In[13]:

#multiple representations
class Number:
        def __add__(self, other):
            return self.add(other)
        def __mul__(self, other):
            return self.mul(other)
class Complex(Number):
        def add(self, other):
            return ComplexRI(self.real + other.real, self.imag + other.imag)
        def mul(self, other):
            magnitude = self.magnitude * other.magnitude
            return ComplexMA(magnitude, self.angle + other.angle)
from math import sin, cos, pi        
from math import atan2
class ComplexRI(Complex):
        def __init__(self, real, imag):
            self.real = real
            self.imag = imag
        @property
        def magnitude(self):
            return (self.real ** 2 + self.imag ** 2) ** 0.5
        @property
        def angle(self):
            return atan2(self.imag, self.real)
        def __repr__(self):
            return 'ComplexRI({0:g}, {1:g})'.format(self.real, self.imag)
class ComplexMA(Complex):
        def __init__(self, magnitude, angle):
            self.magnitude = magnitude
            self.angle = angle
        @property
        def real(self):
            return self.magnitude * cos(self.angle)
        @property
        def imag(self):
            return self.magnitude * sin(self.angle)
        def __repr__(self):
            return 'ComplexMA({0:g}, {1:g} * pi)'.format(self.magnitude, self.angle/pi)
ma = ComplexMA(2, pi/2)
ma.imag
from fractions import gcd
class Rational(Number):
        def __init__(self, numer, denom):
            g = gcd(numer, denom)
            self.numer = numer // g
            self.denom = denom // g
        def __repr__(self):
            return 'Rational({0}, {1})'.format(self.numer, self.denom)
        def add(self, other):
            nx, dx = self.numer, self.denom
            ny, dy = other.numer, other.denom
            return Rational(nx * dy + ny * dx, dx * dy)
        def mul(self, other):
            numer = self.numer * other.numer
            denom = self.denom * other.denom
            return Rational(numer, denom)
def is_real(c):
        """Return whether c is a real number with no imaginary part."""
        if isinstance(c, ComplexRI):
            return c.imag == 0
        elif isinstance(c, ComplexMA):
            return c.angle % pi == 0
#coercion
class Number:
        def __add__(self, other):
            x, y = self.coerce(other)
            return x.add(y)
        def __mul__(self, other):
            x, y = self.coerce(other)
            return x.mul(y)
        def coerce(self, other):
            if self.type_tag == other.type_tag:
                return self, other
            elif (self.type_tag, other.type_tag) in self.coercions:
                return (self.coerce_to(other.type_tag), other)
            elif (other.type_tag, self.type_tag) in self.coercions:
                return (self, other.coerce_to(self.type_tag))
        def coerce_to(self, other_tag):
            coercion_fn = self.coercions[(self.type_tag, other_tag)]
            return coercion_fn(self)
       


# In[14]:

#sets

s={1,2,3,4}
len(s)
s.union({1,5})
s.intersection({6, 5, 4, 3})


# In[15]:

def invert_safe(x):
        try:
            return invert(x)
        except ZeroDivisionError as e:
            return str(e)
#iterables, lazy computation, and generators
class LetterIter:
        """An iterator over letters of the alphabet in ASCII order."""
        def __init__(self, start='a', end='e'):
            self.next_letter = start
            self.end = end
        def __next__(self):
            if self.next_letter == self.end:
                raise StopIteration
            letter = self.next_letter
            self.next_letter = chr(ord(letter)+1)
            return letter
letter_iter = LetterIter()
#iterables
class Letters:
    
    def __init__(self, start='a', end='e'):
            self.start = start
            self.end = end
    def __iter__(self):
            return LetterIter(self.start, self.end)
b_to_k = Letters('b', 'k')
first_iterator = b_to_k.__iter__()
next(first_iterator)
#generators
def letters_generator():
        current = 'a'
        while current <= 'd':
            yield current
            current = chr(ord(current)+1)
letters = letters_generator()
class LettersWithYield:
        def __init__(self, start='a', end='e'):
            self.start = start
            self.end = end
        def __iter__(self):
            next_letter = self.start
            while next_letter < self.end:
                yield next_letter
                next_letter = chr(ord(next_letter)+1)


# In[16]:

#magic functions
len?#help
len??#source code
#tab completion 
get_ipython().magic('paste')
get_ipython().magic('cpaste')
get_ipython().magic('run')
get_ipython().magic('timeit')
In 
Out
print(_)
print(__)
math.sin(2) + math.cos(2);
get_ipython().magic('history -n 1-4')
get_ipython().system('ls')
get_ipython().system('pwd')
#Errors and Debugging
get_ipython().magic('xmode plain')
get_ipython().magic('pdb on')
get_ipython().magic('time')
get_ipython().magic('timeit')
get_ipython().magic('prun')
get_ipython().magic('memit')
get_ipython().magic('mprun')


# In[17]:

#Numpy
import numpy
import numpy as np
numpy.__version__
import array
L=list(range(10))
A=array.array('i',L)
A
np.array([1,4,2,5,3])
np.array([range(i,i+3)for i in [2,4,6]])
np.zeros(10, dtype=int)
np.full((3,5), 3.14)
np.arange(0,20,2)
np.linspace(0,1,5)
np.eye(3)
#numpy attributes
import numpy as np
np.random.seed(0)
x1=np.random.randint(10, size=6)
x2=np.random.randint(10, size=(3,4))
x=np.arange(10)
x[:5]
x[5:]
x[::2]
x[::-1]
#no-copy views
x2_sub=x2[:2, :2]
print(x2_sub)
x2_sub[0,0]=99
x2_sub_copy=x2[:2, :2].copy()
print(x2_sub_copy)
#Rehaping of arrays
grid = np.arange(1, 10).reshape((3, 3))
x=np.array([1,2,3])
x[np.newaxis, :]
x[:,np.newaxis]#column vector via new axis
#Array Concatenation and Splitting 
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
np.concatenate([x, y])
grid = np.array([[1, 2, 3],[4, 5, 6]])
np.concatenate([grid, grid])
#vstack 
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
 [6, 5, 4]])
 # vertically stack the arrays
np.vstack([x, grid])
#hstack
y = np.array([[99],[99]])
np.hstack([grid, y])
#splitting of arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
#hsplit
left, right = np.hsplit(grid, [2])
print(left)
print(right)
#vsplit
upper, lower = np.vsplit(grid, [2])
print(upper)
print(lower)

#Universal functions
import numpy as np 
np.random.seed(0)
def compute_reciprocals(values):
    output=np.empty(len(values))
    for i in range(len(values)):
        output[i]=1.0/values[i]
    return output
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)
np.arange(5) / np.arange(1, 6)
x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)
theta = np.linspace(0, np.pi, 3)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))
    


# In[18]:

M = np.ones((3, 2))
a = np.arange(3)
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
np.count_nonzero(x<6)
np.sum(x<6)
np.any(x<5)

x[x<5]
#Fancy Indexing
ind=[3,7,4]
row=np.array([0,1,2])
col=np.array([2,1,3])
x[row,col]
x[row[:,np.newaxis],col]
x=np.zeros(10)
np.add.at(x,i,1)
print(x)
#Sorting Arrays 
x=np.array([2,1,4,3,5])
np.sort(x)
i=np.argsort(x)
rand=np.random.RandomState(42)
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
 'formats':('U10', 'i4', 'f8')})
print(data.dtype)
data[-1]['name']
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])
data_rec = data.view(np.recarray)
data_rec.age


# In[19]:

#pandas
import pandas
pandas.__version__
import pandas as pd
#series
data=pd.Series([0.25,0.5,0.75,1], index=['a','b','c','d'])
population_dict = {'California': 38332521,
 'Texas': 26448193,
 'New York': 19651127,
 'Florida': 19552860,
 'Illinois': 12882135}
population = pd.Series(population_dict)
population
#dataframes
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
 'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
states = pd.DataFrame({'population': population,
 'area': area})
states
states.index
states.columns
data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=['a', 'b', 'c', 'd'])
data[['a', 'e']]
data.loc[0]
data.iloc[1]
area = pd.Series({'California': 423967, 'Texas': 695662,
 'New York': 141297, 'Florida': 170312,
 'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
 'New York': 19651127, 'Florida': 19552860,
 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
data['density']=data['pop']/data['area']
data.T
data.loc[data.density > 100, ['pop', 'density']]
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
A + B
df = pd.DataFrame(A, columns=list('QRST'))
halfrow = df.iloc[0, ::2]
halfrow
data.isnull()
data[data.notnull()]
data.dropna()
data.fillna(method='ffill')
index = [('California', 2000), ('California', 2010),
 ('New York', 2000), ('New York', 2010),
 ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
 18976457, 19378102,
 20851820, 25145561]
pop = pd.Series(populations, index=index)
index = pd.MultiIndex.from_tuples(index)
pop = pop.reindex(index)
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
 names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
 names=['subject', 'type'])
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data=pd.DataFrame(data, index=index, columns=columns)
x = [[1, 2],[3, 4]]
np.concatenate([x, x], axis=1)
ser1=pd.Series(['A','B','C'], index=[1,2,3])
ser2=pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2], axis='col')
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
#matching column names used as a key 
pd.merge(df1, df2, on='employee')
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
pd.merge(df6, df7, how='inner')#joining on indices
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])
planets.dropna().describe()
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
 'data': range(6)}, columns=['key', 'data'])
df.groupby('key').sum()
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],'data1': range(6),'data2': rng.randint(0, 10, 6)},
columns = ['key', 'data1', 'data2'])
#Aggregation
df.groupby('key').aggregate({'data1': 'min','data2': 'max'})
def filter_func(x):
    return x['data2'].std() > 4
print(df); print(df.groupby('key').std());
print(df.groupby('key').filter(filter_func))
df.groupby('key').transform(lambda x: x - x.mean())
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x
print(df); print(df.groupby('key').apply(norm_by_data2))
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2); print(df2.groupby(mapping).sum())
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

#Pivot Tables
import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')
titanic.pivot_table(index='sex', columns='class',
 aggfunc={'survived':sum, 'fare':'mean'})
births.index = pd.to_datetime(10000 * births.year +
 100 * births.month +
births.day, format='%Y%m%d')

#String operations
data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]
names = pd.Series(data)
names.str.capitalize()
 monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
 'Eric Idle', 'Terry Jones', 'Michael Palin'])
monte.str.extract('([A-Za-z]+)')
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
monte.str.split().str.get(-1)

#Time Series
full_monte = pd.DataFrame({'name': monte,
 'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C',
 'B|C|D']})
full_monte['info'].str.get_dummies('|')
from datetime import datetime
datetime(year=2015, month=7, day=4)

from dateutil import parser
date = parser.parse("4th of July, 2015")
#Indexing
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
 '2015-07-04', '2015-08-04'])
 data = pd.Series([0, 1, 2, 3], index=index)
data
import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
date + np.arange(12)

pd.date_range('2015-07-03', periods=8, freq='H')
pd.timedelta_range(0, periods=9, freq="2H30T")
#Resampling 
from pandas_datareader import data
goog = data.DataReader('GOOG', start='2004', end='2016',
 data_source='google')
goog.head()
goog.plot(alpha=0.5, style='-')
goog.resample('BA').mean().plot(style=':')
goog.asfreq('BA').plot(style='--');
plt.legend(['input', 'resample', 'asfreq'],loc='upper left')
#Time-Shifts
#tshift shifts the index and shift shifts the data
ROI = 100 * (goog.tshift(-365) / goog - 1)
ROI.plot()
plt.ylabel('% Return on Investment');

#Rolling Windows
rolling = goog.rolling(365, center=True)
data = pd.DataFrame({'input': goog,
 'one-year rolling_mean': rolling.mean(),
 'one-year rolling_std': rolling.std()})
ax = data.plot(style=['-', '--', ':'])
ax.lines[0].set_alpha(0.3)

daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'])
plt.ylabel('mean hourly count');


result2 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result2)
Cmean = df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')








# In[2]:

#Machine Learning
import os
import numpy
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

fetch_housing_data()
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
#import MySQLdb
#import sys
#connection = MySQLdb.connect (host = "192.168.1.2", user = "user", passwd = "password, db = "scripting_mysql")
#cursor = connection.cursor ()
#cursor.execute ("select name_first, name_last from address")
#data=cursor.fetchall()
#for row in data:
#    print row[0],row[1]
#cursor.close()
#connection.close()
#sys.exit()
import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
#opening the connection and grabbing the page
my_url="https://www.newegg.com/?FM=1"
uClient=uReq(my_url)
page_html=uClient.read()
uClient.close()
#html parsing
page_soup=soup(page_html, "html.parser")
page_soup.body.span
#grabs each product
containers=page_soup.find_all("div",{"class":"item-container"})
containers[0]
containers.div.div.a.img["title"]
filename="products.csv"
f=open(filename, "w")
headers="brand, product_name, shiping\n"
f.write(headers)
for container in containers:
    brand=container.div.div.a.img["title"]
    title_container=container.findAll("a", {"class":"item-title"})
    product_name=title_container[0].text
    shipping_container=container.findAll("li", {"class":"price-ship"})
    shipping_container[0].text.strip()
    f.write(brand + ","+product_name.replace(",","|")+","+shipping"\n")
f.close()
    

    








# In[21]:

#Exploratory Analysis
import numpy as np
housing=load_housing_data()
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
#create a test set
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts() / len(housing)
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# In[22]:

#Discover and Visualize data 
housing=strat_train_set.copy()
housing.describe()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='population', c="median_house_value", 
            cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
#Looking for Correlations
corr_matrix=housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False) 

from pandas.tools.plotting import scatter_matrix
attributes=["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8)) 
#Attribute combinations ~ caculated metrics
housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#separate the predictors and the target vaariable
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#missing values 
from sklearn.preprocessing import Imputer 
imputer=Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X=imputer.transform(housing_num)
housing_tr=pd.DataFrame(X, columns=housing_num.columns)

#handling Text and categorical attributes
from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot=encoder.fit_transform(housing_cat)
housing_cat_1hot








# In[23]:

#Custom transformers
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix= 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#Transformation pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
 ('imputer', Imputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#Categorical and numerical transformers
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    

from sklearn.pipeline import FeatureUnion
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
 ('selector', DataFrameSelector(num_attribs)),
 ('imputer', Imputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
cat_pipeline = Pipeline([
 ('selector', DataFrameSelector(cat_attribs)),
 ('label_binarizer', LabelBinarizer()),
 ])
full_pipeline = FeatureUnion(transformer_list=[
 ("num_pipeline", num_pipeline),
 ("cat_pipeline", cat_pipeline),
 ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared



# In[24]:

#Select and Train a model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
some_data_prepared=full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))
from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()

#Cross Validation 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Mean:", scores.mean())
display_scores(rmse_scores)

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

#Hyperparameter Tuning
#Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#randomized search(larger number of combinations)

feature_importances=grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

#Evaluate the test set 
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) 





# In[25]:

#Classification
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X,y=mnist["data"], mnist["target"]
X.shape
y.shape
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
 interpolation="nearest")
plt.axis("off")
plt.show()


# In[26]:

y[36000]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#5-detector
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])



# In[27]:

#Performance Measures
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
#precision
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) 
recall_score(y_train_5, y_train_pred)
y_scores = sgd_clf.decision_function([some_digit])
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
 method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
#ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()


# In[28]:

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
 method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()


# In[29]:

#Multiclass Classification
#OVA(default except SVM) vs OVO
sgd_clf.fit(X_train, y_train)
some_digit_scores = sgd_clf.decision_function([some_digit])
np.argmax(some_digit_scores)
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])


# In[30]:

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[31]:

#Multi-label classification
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)


#Multi-Output Classification
noise = rnd.randint(0, 100, (len(X_train), 784))
noise = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)


# In[ ]:

import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
#using scikit learn 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

#Batch Gradient Descent
eta = 0.1 # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2,1) # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
#stochastic gradient descent
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t):
    return t0 / (t + t1)
theta = np.random.randn(2,1) # random initialization
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
#SGD regressor using scikit 
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

#Polynomial regression 
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

#learning Curves
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline((
 ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
 ("sgd_reg", LinearRegression()),
 ))
plot_learning_curves(polynomial_regression, X, y)
#Regularized linear models
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
#lasso Regression 
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])

#Elastic Net
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])

#early stopping
from sklearn.base import clone
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None,
 learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train, y_train) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
        
#Logistic regression
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())
X = iris["data"][:, 3:] 
y = (iris["target"] == 2).astype(np.int) 
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
log_reg.predict([[1.7], [1.5]])

#softmax regression
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
softmax_reg.predict([[5, 2]])
softmax_reg.predict_proba([[5, 2]])



# In[ ]:

#Support Vector Machines
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
svm_clf = Pipeline((
 ("scaler", StandardScaler()),
 ("linear_svc", LinearSVC(C=1, loss="hinge")),
 ))
svm_clf.fit(X_scaled, y)
svm_clf.predict([[5.5, 1.7]])
#polynomial kernel 
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline((
 ("scaler", StandardScaler()),
 ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
 ))
poly_kernel_svm_clf.fit(X, y)
#RBF kernel 
rbf_kernel_svm_clf = Pipeline((
 ("scaler", StandardScaler()),
 ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
 ))
rbf_kernel_svm_clf.fit(X, y)
#SVM Regression 
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
from sklearn.svm import SVR
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)

#Decision Trees
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

#Decision Tree Visualization
from sklearn.tree import export_graphviz
export_graphviz(
 tree_clf,
 out_file=image_path("iris_tree.dot"),
 feature_names=iris.feature_names[2:],
 class_names=iris.target_names,
 rounded=True,
 filled=True
 )
#Decision Tree Regression 
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
#Ensemble Learning and Random Forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
 estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
 voting='hard'
 )
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
 DecisionTreeClassifier(), n_estimators=500,
 max_samples=100, bootstrap=True, n_jobs=-1
 )
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
#feature importance
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)

#Boosting
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
 DecisionTreeClassifier(max_depth=1), n_estimators=200,
 algorithm="SAMME.R", learning_rate=0.5
 )
ada_clf.fit(X_train, y_train)

#Gradient Boosting
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_val, y_train, y_val = train_test_split(X, y)
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)
errors = [mean_squared_error(y_val, y_pred)
 for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors)
gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators)
gbrt_best.fit(X_train, y_train)

#Early Stopping
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
        break # early stopping
#Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>=0.95)+1
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
#Incremental PCA
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_mnist, n_batches):
    inc_pca.partial_fit(X_batch)
X_mnist_reduced = inc_pca.transform(X_mnist)
#Kernal PCA
from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
clf = Pipeline([
 ("kpca", KernelPCA(n_components=2)),
 ("log_reg", LogisticRegression())
 ])
param_grid = [{
 "kpca__gamma": np.linspace(0.03, 0.05, 10),
 "kpca__kernel": ["rbf", "sigmoid"]
 }]
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
#RBF PCA
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
from sklearn.metrics import mean_squared_error
mean_squared_error(X, X_preimage)
print pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2'])

#LLE
from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

        

