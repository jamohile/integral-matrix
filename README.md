# Integral Matrix

This is a simple implementation of the summed-area table used by Frank Crow, Viola-Jones Algorithm, etc.

I found the approach really elegant, and wanted to explore a quick implementation.

Run Tests: (from root)

`python -m unittest discover tests`

Motivation:
- There are some cases where you need to compute a number of region-sums in a matrix. Done naively through iteration, this can be slow and expensive.

Approach:
- Precompute running totals for each element (the integral table)
- To find the sum of a given region, use the following approach:

```
a b c
d e f
g h i

Sum (e...i) = sum(a...i) - sum(a...g) - sum(a...c) + sum(a)

(Where the last sum(a) is because it was double-counted in the subtraction.)
```

If we are alright with discarding the original matrix, this can be done with no additional space, otherwise, it doubles space requirements.

As a tradeoff, we can calculate any MxN region-sum in O(1) time rather than O(MN).