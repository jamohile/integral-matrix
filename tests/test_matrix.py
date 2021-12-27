import unittest
import numpy as np

from matrix import IntegralMatrix


class MatrixTests(unittest.TestCase):
    def test_correct_small_matrix(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2],
                [3, 4]
            ]))
        integral = im.get_integral()

        np.testing.assert_equal(integral,
                                [
                                    [1, 3], [4, 10]
                                ])

    def test_correct_horizontal_matrix(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6]
            ]))
        integral = im.get_integral()

        np.testing.assert_equal(integral,
                                [
                                    [1, 3, 6],
                                    [5, 12, 21]
                                ])

    def test_correct_vertical_matrix(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6]
            ]))
        integral = im.get_integral()

        np.testing.assert_equal(integral,
                                [
                                    [1, 3],
                                    [4, 10],
                                    [9, 21]
                                ])

    def test_correct_3x3_matrix(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))
        integral = im.get_integral()

        np.testing.assert_equal(integral,
                                [
                                    [1, 3, 6],
                                    [5, 12, 21],
                                    [12, 27, 45]
                                ])

    def test_raises_on_empty_matrix(self):
        self.assertRaises(AssertionError, lambda: IntegralMatrix(np.array([])))

    def test_calculate_region_integral(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))

        # Calculate integral on bottom right region: [5, 6], [8, 9]
        integral = im.get_region_integral(1, 1, 2, 2)
        self.assertEqual(integral, 28)

    def test_calculate_nonsquare_region_integral(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))

        # Calculate integral on right region: [2, 3], [5, 6], [8, 9]
        integral = im.get_region_integral(1, 0, 2, 3)
        self.assertEqual(integral, 33)

    def test_calculate_empty_region_integral(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))

        integral = im.get_region_integral(0, 0, 0, 0)
        self.assertEqual(integral, 0)

    def test_calculate_single_region_integral(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))

        integral = im.get_region_integral(0, 0, 1, 1)
        self.assertEqual(integral, 1)

    def test_raises_if_x_out_of_bounds(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))
        self.assertRaises(
            IndexError, lambda: im.get_region_integral(3, 0, 1, 1))

    def test_raises_if_y_out_of_bounds(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))
        self.assertRaises(
            IndexError, lambda: im.get_region_integral(0, 3, 1, 1))

    def test_raises_if_x_end_out_of_bounds(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))
        self.assertRaises(
            IndexError, lambda: im.get_region_integral(0, 0, 4, 1))

    def test_raises_if_y_end_out_of_bounds(self):
        im = IntegralMatrix(np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]))
        self.assertRaises(
            IndexError, lambda: im.get_region_integral(0, 0, 1, 4))
