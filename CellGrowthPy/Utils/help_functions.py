#Why is this not in base python
def mean(numbers):
    '''

    '''
    return float(sum(numbers)) / max(len(numbers), 1)

def str_to_bool(s):
    if s == 'True' or s == True:
         return True
    elif s == 'False' or s == False:
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

def str_none_check(s):
    if s == 'None' or s == None:
         return None
    else:
         return s