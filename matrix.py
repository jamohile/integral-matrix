import numpy as np


class IntegralMatrix:
    integral_matrix: np.ndarray

    def __init__(self, input_matrix: np.ndarray):
        # For now, only 2D is supported.
        assert len(input_matrix.shape) == 2

        # We store a matrix one row and column larger than the input.
        # This makes it easier to handle operations at leading edges.
        input_shape = input_matrix.shape

        self.integral_matrix = np.zeros(
            (input_shape[0] + 1, input_shape[1] + 1))
        self.integral_matrix[1:, 1:] = input_matrix

        self._populate_integral()

    def get_integral(self) -> np.ndarray:
        return self.integral_matrix[1:, 1:]

    def _populate_integral(self):
        for j in range(1, self.integral_matrix.shape[0]):
            for i in range(1, self.integral_matrix.shape[1]):
                left_integral = self.integral_matrix[j, i-1]
                top_integral = self.integral_matrix[j-1, i]
                top_left_integral = self.integral_matrix[j-1, i-1]

                integral = self.integral_matrix[j, i] + \
                    left_integral + top_integral - top_left_integral
                self.integral_matrix[j, i] = integral
    
    def get_region_integral(self, x, y, size_x, size_y) -> float:
        # Although X, Y are relative to the user's matrix, we treat end x/y relative to
        # internal 0-padded matrix.
        end_x = x + size_x
        end_y = y + size_y

        # Make sure that bounds are all OK.
        assert(0 <= x <= end_x <= (self.integral_matrix.shape[0]))
        assert(0 <= y <= end_y <= (self.integral_matrix.shape[1]))

        # Now, we can also use the passed in x/y for the BEFORE left, top that we need.
        bottom_right_integral = self.integral_matrix[end_y, end_x]
        
        left_integral = self.integral_matrix[end_y, x]
        top_integral = self.integral_matrix[y, end_x]

        top_left_integral = self.integral_matrix[x, y]

        return bottom_right_integral - left_integral - top_integral + top_left_integral


