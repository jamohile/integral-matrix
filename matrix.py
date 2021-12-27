import numpy as np


class IntegralMatrix:
    """
    IntegralMatrix allows fast computation of region-sums within a 2D Matrix.
    It allows calculation of an MxN region in O(1) rather than O(MN) time (at the cost of O(total_width, total_height) memory)

    This is an implementation of the Summed Area Table (Frank Crow) or Integral Image (Viola-Jones).
    """

    integral_matrix: np.ndarray

    def __init__(self, input_matrix: np.ndarray):
        """
        Args:
            input_matrix: 2D numpy matrix we'd like to calculate regions on.
        """

        # For now, only 2D is supported.
        assert len(input_matrix.shape) == 2

        # We store a matrix one row and column larger than the input.
        # This makes it easier to handle operations at leading edges.
        input_shape = input_matrix.shape
        integral_shape = (input_shape[0] + 1, input_shape[1] + 1)

        self.integral_matrix = np.zeros(integral_shape)
        
        # Leave a col/row of leading 0s in the integral matrix.
        self.integral_matrix[1:, 1:] = input_matrix

        self._populate_integral()

    def get_integral(self) -> np.ndarray:
        return self.integral_matrix[1:, 1:]

    def _populate_integral(self):
        """
        Populate the integral matrix by pre-calculating running totals at each index.
        That is, element(i, j) = sum(all elements in the region (0, 0) -> this element)
        """
        
        for j in range(1, self.integral_matrix.shape[0]):
            for i in range(1, self.integral_matrix.shape[1]):
                # We add together the running totals on left, and on top
                # (which are complete because of the iteration order)
                # Then subtract the portion that is double counted between them.

                left_integral = self.integral_matrix[j, i-1]
                top_integral = self.integral_matrix[j-1, i]
                top_left_integral = self.integral_matrix[j-1, i-1]

                integral = self.integral_matrix[j, i] + \
                    left_integral + top_integral - top_left_integral
                self.integral_matrix[j, i] = integral

    def get_region_integral(self, x, y, size_x, size_y) -> float:
        """
        Calculate the sum of a region in our input-matrix.
        For example, given input:
            [a b c d]
            [e f g h]
            [i j k l]
            [m n o p]
        get_region_integral(1, 1, 2, 2) -> sum(f, g, j, k)
        """

        # Note that these are technically 1 larger than they should be.
        # E.g, x coordinate of 1 with size 1 -> 2.
        # This is OK, since internally integral_matrix is 0-padded on top/left.
        end_x = x + size_x
        end_y = y + size_y

        # Make sure that bounds are all OK.
        assert(0 <= x <= end_x <= (self.integral_matrix.shape[0]))
        assert(0 <= y <= end_y <= (self.integral_matrix.shape[1]))

        """
        We use the following approach.
        Say we want sum in region: [[f, g], [j, k]]

        1. Get bottom_right sum:
            [a...c]
            [. . .]
            [i...k]

        2. Subtract left_sum:
            [a]
            [.]
            [i]

        3. Subtract top_sum:
            [a...c]
        
        4. Re-add double subtracted top_left sum:
            [a]

        Each of these sums are pre-computed in the integral matrix.
        So, we can compute an MxN region in O(1) rather than O(MN) time.
        """

        bottom_right_integral = self.integral_matrix[end_y, end_x]

        # Note that when we calculate these, we want the portions *before* the region.
        # Since x, y are offset by 1 due to the top/left padding, x/y on their own already represent
        # these before-portions.
        left_integral = self.integral_matrix[end_y, x]
        top_integral = self.integral_matrix[y, end_x]

        top_left_integral = self.integral_matrix[x, y]

        return bottom_right_integral - left_integral - top_integral + top_left_integral
