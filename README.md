## pmet

Library for classification and metric learning on strings.

### Motivation

#### Classification

An example of when string classification might be useful is when we want to build a model that can determine the "semantic type" of a string.  That is, a function like:

```

    >>> f("1600 Pennsylvania Ave")
    "address"
    
    >>> f("Kurt Russell")
    "name"
    
    >>> f("ben@aol.com")
    "email_address"

    >>> f("1-321-345-1234")
    "phone_number"
    
    >>> f("964-32-6523")
    "ssn"

```

We're calling it "semantic type" because it goes beyond `[int, float, string, bool, ...]`, towards the actual _meaning_ of the value. 

It automated data cleaning, running this kind of function over the columns of a dataset is helpful in determining what kind of information a dataset might contain, and how it's contents might align with another dataset.

#### Metric learning

Classification works when we have a closed set of labels -- that is, we know the labels that we care about ahead of time, and we have examples of each of the classes.  Metric learning can help in a situation where the only kinds of labels we have are "this set of things are allsimilar" and "these sets of strings are dissimilar".