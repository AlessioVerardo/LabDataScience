# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Python Refresher
#
# The purpose of this notebook is to get some practice with the special python concepts that are useful for writing Spark applications. Here we will do them in the (safe) controlled environment of a simple python shell for simpler debugging -- later on we will use these same constructs in the Spark framework.
#
# ## How to use this notebook
# You can (and should) execute all the cells in this notebook. Where your input is required, you will see <FILL> in the source code with some instructions. Replace those with working code, execute, debug, rinse, repeat.
#
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Outline
#
# #### 0. [Notebook demo](#Notebook-intro)
# #### 1. [Python Datatypes](#Datatypes)
# #### 2. [Functional programming and MapReduce](#Map-reduce)
# #### 3. [Lambda functions](#"Lambda"-functions)
# #### 4. [List comprehensions](#List-comprehension) (and [Tuples](#Tuples))
# #### 5. [Generator expressions](#Generator-expressions)
# #### 6. [Generators](#Generators)
# #### 7. [Final exercices](#Exercise)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Notebook intro
#
# This is a Jupyter notebook... you can use it to, for example, run some python code...
#
# #### To run the code below:
#
# 1. Click on the cell to select it.
# 2. Press <kbd>SHIFT+ENTER</kbd> on your keyboard or press the play button in the toolbar above.
#
# Before going on to work on the `python` and `spark` tutorials yourself, do check out the [user interface tutorial](http://nbviewer.ipython.org/github/ipython/ipython/blob/2.x/examples/Notebook/User%20Interface.ipynb).
#

# %% slideshow={"slide_type": "subslide"}
a = 1
b = 2
a+b

# %% [markdown] slideshow={"slide_type": "subslide"}
# Some of the most important keyboard shortcuts to make your life easier: 
#
# * to insert a new cell below the current cell, push <kbd>b</kbd>. 
# * to insert a cell above, push <kbd>a</kbd> 
# * to delete a cell, push <kbd>d</kbd> *twice*
# * to undo a cell operation, push <kbd>z</kbd>
# * to change a cell to "markdown" (i.e. text, like this one) push <kbd>m</kbd>
# * to change a cell to "code" (i.e. to run python code, like the cell above) push <kbd>y</kbd>
#
# Try it!
#
# To see the keyboard shortcuts, click on "Help" in the toolbar and select "Keyboard Shortcuts"

# %% [markdown] slideshow={"slide_type": "slide"}
# #### Ok, now we're ready to continue...

# %% [markdown] slideshow={"slide_type": "skip"}
# ## Datatypes
#
# This is a quick primer on various high-level python data types that we will be using. If you are not familiar with these at least superficially already, we recommend you first find a python tutorial of some sort. 

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Lists
#
# A list is just that -- an ordered collection of objects. You can put just about any python object into a list. Most likely, we will be dealing with lists of arrays, lists of tuples, lists of dictionaries, etc. A list is really an essential building block!

# %% slideshow={"slide_type": "skip"}
# a list of integers
my_list = [1,2,3,4]

# %% [markdown] slideshow={"slide_type": "skip"}
# The methods of a `list` object are not many, but they are quite useful. 
#
# add an empty cell below this one, and type 
#
# ```
# my_list.
# ```
#
# followed by tapping the <kbd>TAB</kbd> key to see the list of methods. 
#

# %% slideshow={"slide_type": "skip"}
# TODO: append the value 1 to my_list, making it [1,2,3,4,1]
my_list.append(1)
assert(len(my_list) == 5 and sum(my_list) == 11)

# %% [markdown] slideshow={"slide_type": "skip"}
# #### Indexing 
# List indexing tricks will be essential to your experience with any sort of application in Spark. Here are a few of the most common ones: 

# %% slideshow={"slide_type": "skip"}
# slices --> getting consecutive items from a list
my_list[2:5]

# %% slideshow={"slide_type": "skip"}
# all elements from the first to the third
my_list[:3] # zero is implied

# %% slideshow={"slide_type": "skip"}
# reversing 
my_list[::-1]

# %% slideshow={"slide_type": "skip"}
# skipping elements --> here, getting every other one
my_list[::2]

# %% slideshow={"slide_type": "skip"}
# getting the second to last element
my_list[-2]

# %% slideshow={"slide_type": "skip"}
# TODO: make a new_list composed of all elements from my_list except for the first and last one
new_list = my_list[1:-1]
assert(new_list == [2,3,4])

# %% slideshow={"slide_type": "skip"}
# combining lists 
my_list + new_list

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Dictionaries
# Certainly one of the most useful built-in data structures in python. A dictionary provides a mapping between "keys" and "values". Best to look at some examples. 

# %% slideshow={"slide_type": "skip"}
# creating a dictionary
d = {} # makes an empty dictionary
type(d) 

# %% slideshow={"slide_type": "skip"}
# add an element
d['first'] = 1
d['second'] = 2
d

# %% slideshow={"slide_type": "skip"}
d['first'], d['second']

# %% slideshow={"slide_type": "skip"}
# TODO: iterate through all the keys of the dictionary "d" and print out its values 
# HINT: type "d." and tap the "tab" key to see the available methods 
#       or lookup the python dictionary documentation
for k in d.keys() : 
    print(k)

# %% [markdown] slideshow={"slide_type": "skip"}
# Dictionaries have some very useful methods: 

# %% slideshow={"slide_type": "skip"}
# to see all the currently stored keys or values
print(d.keys())
print(d.values())

# %% slideshow={"slide_type": "skip"}
# alternative way of initializing a dictionary: 
d = {'first': 1, 'second': 2}
print(d)
d['third'] = 3
print(d)

# %% [markdown] slideshow={"slide_type": "skip"}
# <div class="alert alert-info">
# <p><strong>Note</strong></p> 
#
# <p>Do not always assume that the keys from a dictionary are returned in the order they are entered! (Unless you use an <a href="https://docs.python.org/3/library/collections.html#collections.OrderedDict">OrderedDict</a>.)</p>
# </div>

# %% [markdown] slideshow={"slide_type": "skip"}
# ### Strings
# Not a complex data type really, but strings are objects like all other things in python, and have some nice properties. 

# %% slideshow={"slide_type": "skip"}
# they can be indexed like any other collection
string = 'what is going on here'
print('second to seventh characteres: "%s"'%string[2:7])
print('last character: "%s"'%string[-1])

# %% [markdown] slideshow={"slide_type": "skip"}
# ... you get the idea... 

# %% slideshow={"slide_type": "skip"}
# getting words from a string
string2 = 'one,two,three,four'
print(string.split())    # default splits on whitespaces
print(string2.split(',')) # but you can specify any delimiter you want

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Map-reduce 
#
# The map-reduce programming model is at the heart of distributed data processing. In essence, it is quite simple: 
#
# 1. start with a collection of data and distribute it
# 2. define a function you want to use to operate on that data
# 2. apply the function to every element in the data collection (the *map* step)
# 3. once the data has been massaged into a useful state, compute some aggregate value and return it to the user (the *reduce* step)

# %% [markdown] slideshow={"slide_type": "subslide"}
# A few things to note: 
#
# 1. this is an extremely limiting programming model (compare to MPI where anything is possible)
# 2. from user's viewpoint strictly task-parallel --> can't make tasks communicate to each other 
# 3. very clear on intent *because* it is so limiting
#
# Let's see how this works through a simple example. 

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Very very basic MapReduce example
#
# First, we define our data array, in this case we're not very creative and just use 10 random integers in the range 0 - 100:

# %% slideshow={"slide_type": "-"}
import random
random.seed(1) # initialized to make sure we get the same numbers every time
data = []
for x in range(10) : data.append(random.randint(0,100))
print(data)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Lets say that we wanted to compute the total sum of all the values after applying some function $f(x)$ to them. We'll say for now that $$f(x) = 2x.$$ The most obvious choice for this would be to apply $f(x)$ in some sort of a loop, and add the results to an aggregation variable: 

# %% slideshow={"slide_type": "-"}
dbl_sum = 0
for x in data : 
    dbl_sum += x*2
    
print(dbl_sum)


# %% [markdown] slideshow={"slide_type": "fragment"}
# In this case, the calculation was entirely sequential:
#
# we went through each element in `data`, doubled it, and added the result to the aggregate variable `dbl_sum` all in a single step. 

# %% [markdown] slideshow={"slide_type": "subslide"}
# But the two stages are separable: 
#
# we might *first* double all the elements in `data` - apply $f(x)$ - and then sum them all together. 
#
# This is exactly a map-reduce operation: 
#
# 1. *map* the values using a function ($f(x) = 2x$) 
# 2. *reduce* them to a single number (sum) 

# %% [markdown] slideshow={"slide_type": "subslide"}
# As it turns out, the `python` language already includes the `map` and `reduce` functions so we can try this out immediately. First, we define the function that will be used as a `map`:

# %% slideshow={"slide_type": "fragment"}
def double_the_number(x) : 
    return x*2


# %% [markdown] slideshow={"slide_type": "fragment"}
# Now we apply the `map` -- notice how compact this looks!

# %%
dbl_data = map(double_the_number, data)
print(dbl_data)

# %% [markdown]
# `map` implicitly loops over all of the elements of data and applies `double_the_number` to each one. 

# %% [markdown] slideshow={"slide_type": "subslide"}
# For the reduction, we will use the standard `add` operator: 

# %%
from operator import add
from functools import reduce
reduce(add, dbl_data)

# %% [markdown] slideshow={"slide_type": "subslide"}
# <div class="alert alert-info" style="margin: 10px"><strong>Note:</strong> what we are doing here is *functional* programming - we use a function to transform the data, but the original data remains untouched. The Spark programming model is heavily based on this concept. 
# </div>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## "Lambda" functions
#
# * our function `double_the_number` needed a lot of writing for a very simple operation 
# * but! `map` *requires* a function to apply to the data array
# * when a function needed is very simple, the concept of "in-line" lambda functions is great 

# %% [markdown] slideshow={"slide_type": "subslide"}
# Basic idea: 
#
# * the lambda function consumes items and returns a value
# * it can get items from an iterable (a list, dictionary, tuple, etc.)
# * it returns one element for each element it takes in  
#
# Here are two simple examples:

# %% slideshow={"slide_type": "fragment"}
double_lambda = lambda x: x*2
double_lambda(4)

# %%
add_two_numbers = lambda x,y: x+y
add_two_numbers(1,2)

# %% [markdown] slideshow={"slide_type": "subslide"}
# We can use our `lambda` function as the function we pass to `map`:

# %% slideshow={"slide_type": "-"}
dbl_data = list(map(lambda x: x*2, data))
dbl_data

# %% [markdown]
# This form has the advantage of being much more compact and allowing function creation "on the fly". 
#
# The concept of in-line functions will be key to writing simple Spark applications!

# %% [markdown] slideshow={"slide_type": "subslide"}
# Note that a `lambda` function is a function just like any other and you can also give it a name (although tha almost defies the point of an in-line function...)

# %%
double_lambda = lambda x: x*2
print('type of double_lambda is %s ' % type(double_lambda))

# %% [markdown] slideshow={"slide_type": "slide"}
# ## List comprehension
#
# "List comprehension" is a complicated name for a pretty nice feature of python: creating lists on the fly using any kind of iterable object, often with the help of lambda functions. 
#
# * In many cases, a handy replacement for `for` loops when creating lists of objects 
# * can sometimes perform faster than the equivalent for loop 

# %% [markdown] slideshow={"slide_type": "subslide"}
# A normal python list is made by 

# %% slideshow={"slide_type": "-"}
my_list = [1, 2, 3, 4, 5]

# %% [markdown] slideshow={"slide_type": "subslide"}
# The basic syntax for a *list comprehension* is that you enclose a `for` loop *inside* the list brackets `[]`. 

# %% [markdown] slideshow={"slide_type": "fragment"}
# To make a simple (slightly contrived) example, consider: 

# %% slideshow={"slide_type": "-"}
simple_list = [x for x in my_list]
simple_list

# %% [markdown]
# (should really be called list *expansion*)

# %% [markdown] slideshow={"slide_type": "fragment"}
# The equivalent `for` loop:

# %%
simple_list = []
for x in my_list: 
    simple_list.append(x)
simple_list

# %% [markdown] slideshow={"slide_type": "subslide"}
# The list comprehension gives you much more concise syntax!
#
# Even neater when a conditional is used in the iteration: 

# %%
# only even numbers
simple_list = []
for x in my_list: 
    if x % 2 == 0:
        simple_list.append(x)
simple_list

# %%
[x for x in my_list if x % 2 == 0]

# %% [markdown] slideshow={"slide_type": "slide"}
# <div class="alert alert-warning"> 
# The construct 
#
#     `f(x) for x in y`
#     
# is *extremely* powerful! 
#
# Anything that can be iterated can be used as the `y`. In the case above, `f(x) = x`, but it could be any function you want (including of course a lambda function!) 
# </div>

# %% [markdown] slideshow={"slide_type": "subslide"}
#
# ### Tuples
# Lets make a simple list of tuples to see one common application of such list comprehensions: 

# %% slideshow={"slide_type": "-"}
tuple_list = list(zip([1,2,3,4], ['a','b','c','d']))

# %% [markdown] slideshow={"slide_type": "fragment"}
# Now we want to extract just the letters out of this list:

# %%
[x[1] for x in tuple_list]

# %% [markdown] slideshow={"slide_type": "fragment"}
# An even clearer syntax is to label the tuple elements that we are extracting from the list:

# %%
[letter for (num, letter) in tuple_list]

# %% [markdown]
# This notation is very elegant and allows us to do a reasonably complex operation (iterate over the list and extracting elements of a tuple into a new list) in a very simple way. 

# %% [markdown] slideshow={"slide_type": "subslide"}
# A conditional can be applied on the values in the iterator when creating the new list when processing a tuple, just as we did above.

# %% [markdown] slideshow={"slide_type": "-"}
# For example, if we wanted only the letters corresponding to all the even values: 

# %%
[letter for (num, letter) in tuple_list if num%2 == 0]


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Filter
#
# Sometimes some more complex logic needs to be applied to the values for filtering. 
#
# * for such cases, use the `filter` function
# * can be any function of the form `f(x) --> boolean` - even a lambda!

# %%
def filter_func(x) :
    num, letter = x
    return num%2 == 0

filtered_tuple_list = filter(filter_func, tuple_list)
list(filtered_tuple_list)

# %% [markdown] slideshow={"slide_type": "fragment"}
# We can of course use the results of `filter` also in a list comprehension:

# %%
[letter for (num,letter) in filter(filter_func, tuple_list)]

# %% [markdown] slideshow={"slide_type": "subslide"}
# Slight **warning** here: 
#
# the elements of the list are tuples and the function arguments don't expand the tuple automatically. That's why we have the extra line
#
#     num, letter = x
#
# which takes `num` and `letter` out of each tuple that gets passed to `filter_func`. 
#
# The same would happen with a `lambda funcion`: 

# %%
# error
list(filter(lambda x,y: x<0, tuple_list))

# %% [markdown] slideshow={"slide_type": "skip"}
# ### To-do:
#
# Rewrite `filter_func` as a lambda function with the correct syntax:

# %% slideshow={"slide_type": "skip"}
[letter for (num, letter) in filter(lambda x: x[0]%2 == 0, tuple_list)]

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Generator expressions 
#
# Unfortunately, creating long lists can have large memory overhead. 
#
# Often, we don't need to hold the entire lists in memory, but only need the elements one by one -- this is the case with *all* reductions, for example, such as the `sum` we used above. 

# %% [markdown] slideshow={"slide_type": "subslide"}
# In the cell below, two lists are actually created -- first, the one returned by `range` and once this one is iterated over, we have a second list resulting from the `x for x in range` part:

# %%
sum([x for x in range(1000000)])

# %% [markdown] slideshow={"slide_type": "subslide"}
# When dealing with large amounts of data, the memory footprint becomes a serious concern and can make a difference between a code completing or crashing with an "out of memory" error. 
#
# Luckily, `python` has a neat solution for this, and it's called "generator expressions". The gist is that such an expression acts like an **iterable**, but only creates the items when they are requested, computing them on the fly. 

# %% [markdown] slideshow={"slide_type": "subslide"}
# Generator expressions work *exactly* the same way as list comprehension, but using `()` instead of `[]`. Very nice. 
#
# So, lets see how this works: 

# %%
print(sum(x for x in range(1000000)))

# now summing only the even numbers -- conditionals work just like in list comprehension
print(sum(x for x in range(1000000) if x % 2 == 0))

# %% [markdown] slideshow={"slide_type": "subslide"}
# The downside is that the elements of a generator expression can be accessed exactly once, i.e. there is *no* indexing!

# %%
list_expression = [x for x in range(100)]
list_expression[5]

# %% slideshow={"slide_type": "fragment"}
gen_expression = [x for x in range(100)]
gen_expression[5]


# %% [markdown] slideshow={"slide_type": "subslide"}
# Finally, because creating long lists of integers, e.g. `range`, is so common and so wasteful, the `python3` version of `range` actually acts like a generator -- instead of making a list, this simply yields the elements one by one. 
#
# Compare memory usage of these two executions: 

# %%
# %load_ext memory_profiler
# %memit sum([x for x in range(10000000)])

# %%
# %memit sum(x for x in range(10000000))

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Generators
# Closely related to generator *expressions* are *generators* - they are 
#
# * functions that keep track of their internal state when they return 
# * on next call they continue from where they left off. 
#
# It's easy to illustrate this with writing our own version of the built-in `range`function.

# %% slideshow={"slide_type": "subslide"}
def my_range(N) :
    i = 0
    while i < N :
        yield i
        i += 1


# %%
gen = my_range(10)
gen

# %%
print('first value', next(gen))
print('next value', next(gen))

# %%
[x for x in gen]

# %%
# exhausted iterator
next(gen)

# %% [markdown] slideshow={"slide_type": "subslide"}
# This only scratches the surface of generator functionality in `python`, but for our purposes it is enough. For a more complete discussion see e.g. [the python wiki](https://wiki.python.org/moin/Generators) and [this pretty good example](http://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/). 
#
# Generators and generator expressions are useful in general when dealing with large data objects because they allow you to iterate through the data without ever holding it in memory. 
#
# The concept of generators will be useful when we discuss the `mapPartitions` RDD method in Spark.

# %% [markdown] slideshow={"slide_type": "skip"}
# ## Exercise
#
# Use the python `map` function to convert the first and last letters of each word in the string `suntzu` (defined below) to uppercase.
#
# <div class="alert alert-info">
# <p><strong>hint</strong></p> 
#
# <p>Use the standard string method `split` to create a list of words; then use a `map` to convert the appropriate letters of each word<p>
# </div>
#
# <div class="alert alert-info">
# <p><strong>hint \#2</strong></p> 
#
# <p>Use `Edit -> Split Cell` to create easily-executable code chunks that you can debug. When they all run individually, you can merge them back together.</p>
# </div>

# %% slideshow={"slide_type": "skip"}
# From Sun Tzu's Art of War
suntzu = 'The supreme art of war is to subdue the enemy without fighting.'

words = suntzu.split()

def first_last_capitalize(word) : 
    # first convert the string `word` to a list of characters
    l = list(word)
    
    # now change the first and last character to uppercase (use the upper() method of a string)
    l[0] = l[0].upper()
    l[-1] = l[-1].upper()
    
    # convert back to a string
    return str("".join(l))

upper_lower = map(first_last_capitalize, words)

result = " ".join(upper_lower)
print(result)
assert(result == 'ThE SupremE ArT OF WaR IS TO SubduE ThE EnemY WithouT Fighting.')

# %% [markdown] slideshow={"slide_type": "skip"}
# Use a list comprehension to convert the list of words into a list of tuples, where the first element of the tuple is the word and the second element is the word length.
#
# <div class="alert alert-info">
# <p><strong>hint</strong></p> 
# Use the python built-in `len()` function to get the string length

# %% slideshow={"slide_type": "skip"}
word_length = [ (x,len(x)) for x in words ]

# %% slideshow={"slide_type": "skip"}
print(word_length)
assert(word_length == [('The', 3),
 ('supreme', 7),
 ('art', 3),
 ('of', 2),
 ('war', 3),
 ('is', 2),
 ('to', 2),
 ('subdue', 6),
 ('the', 3),
 ('enemy', 5),
 ('without', 7),
 ('fighting.', 9)])

# %% [markdown] slideshow={"slide_type": "skip"}
# Compute the average word length in the sentence by: 
#
# 1. mapping the `word_length` list from above to contain just the word lengths
# 2. using `reduce` to sum up the lengths
# 3. dividing by the total number of words

# %% slideshow={"slide_type": "skip"}
word_counts = len(words)

# %% slideshow={"slide_type": "skip"}
total_chars = reduce(add,map(lambda x : x[1], word_length))

# %% slideshow={"slide_type": "skip"}
import numpy as np
print(float(total_chars)/len(words))
assert(np.allclose(float(total_chars)/len(words),4.33333333333))

# %% [markdown] slideshow={"slide_type": "skip"}
# Write a generator that returns the next word in the sequence with an even number of characters, using the `suntzu` string defined above. At least two possible solutions!

# %% [markdown]
# **Solution 1:**

# %% slideshow={"slide_type": "skip"}
even_gen = (x for x in suntzu.split() if len(x)%2==0)

# %% slideshow={"slide_type": "skip"}
assert(list(even_gen) == ['of', 'is', 'to', 'subdue'])


# %% [markdown]
# **Solution 2:**

# %%
def gen_even(sentence):
    for word in sentence.split():
        if (len(word)%2 == 0):
            yield word
even_gen = gen_even(suntzu)

# %%
assert(list(even_gen) == ['of', 'is', 'to', 'subdue'])

# %%
