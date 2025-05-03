# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import operator

from absl.testing import absltest
from absl.testing import parameterized
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import tree_math as tm
from tree_math._src import test_util

from tree_math.matrix import leaf_pytree

# pylint: disable=g-complex-comprehension


class MatrixTest(test_util.TestCase):

  def test_matrix(self):
    tree = {
        "a": {"c": jnp.array([0], dtype=jnp.int32), "d": 1},
        "b": {"c": jnp.array([[2]], dtype=jnp.int32), "d": jnp.array([3], dtype=jnp.int32)}
    }
    matrix = tm.Matrix(tree, jax.tree.structure({"a": 0, "b": 0}), [0, 1])
    self.assertEqual(matrix.size, 9)
    self.assertLen(matrix, 3)
    self.assertEqual(matrix.shape, (3, 3))
    self.assertEqual(matrix.ndim, 2)
    self.assertEqual(matrix.dtype, jnp.int32)
    self.assertEqual(repr(tm.Matrix({"a": 1}, leaf_pytree, [0])),
                     "tree_math.Matrix({'a': 1})")
    self.assertTreeEqual(tree_util.tree_leaves(tree),
                         tree_util.tree_leaves(matrix), check_dtypes=True)
    vector2 = tree_util.tree_map(lambda x: x, matrix)
    self.assertTreeEqual(matrix, vector2, check_dtypes=True)

  def test_matmul_scalars(self):
    actual = tm.Matrix(1.0, leaf_pytree, [0]) @ tm.Matrix(2.0, leaf_pytree, [0])
    expected = 2.0
    self.assertAllClose(actual, expected)

  def test_matmul(self):
    rng = np.random.default_rng(0)
    rnorm1 = rng.standard_normal(9, dtype=np.float32)
    rnorm2 = rng.standard_normal(9, dtype=np.float32)
    tree1 = {
        "a": {"c": rnorm1[0] , "d": rnorm1[1:3]},
        "b": {"c": rnorm1[3:5], "d": rnorm1[5:].reshape(2, 2))}
    }
    tree2 = {
        "c": {"e": rnorm2[0] , "f": rnorm2[1:3]},
        "d": {"e": rnorm2[3:5], "f": rnorm2[5:].reshape(2, 2))}
    }
    rng = np.random.default_rng(0)
    tree1 = {"a": rng.standard_normal(dtype=np.float32),
             "b": rng.standard_normal((2, 3), dtype=np.float32)}
    tree2 = {"a": rng.standard_normal(dtype=np.float32),
             "b": rng.standard_normal((2, 3), dtype=np.float32)}

    expected = {
        "a": {
            "e": tree1["a"]["c"] * tree2["c"]["e"] + tree1["a"]["d"] @ tree2["d"]["e"],
            "f": tree1["a"]["c"] * tree2["c"]["f"] + tree1["a"]["d"] @ tree2["d"]["f"]
        },
        "b": {
            "e": tree1["b"]["c"] * tree2["c"]["e"] + tree1["b"]["d"] @ tree2["d"]["e"],
            "f": np.outer(tree1["b"]["c"], tree2["c"]["f"] + tree1["b"]["d"] @ tree2["d"]["f"]
        }
    }
    outer_treedef1 = jax.tree.structure({"a": 0, "b": 0})
    outer_treedef2 = jax.tree.structure({"c": 0, "d": 0})
    matrix1 = tm.Matrix(tree1, outer_treedef1, [0, 1])
    matrix2 = tm.Matrix(tree2, outer_treedef2, [0, 1])

    actual = matrix1 @ matrix2
    self.assertAllClose(actual, expected)

    # actual = vector1.dot(vector2)
    # self.assertAllClose(actual, expected)

    with self.assertRaisesRegex(
        TypeError,
        "matmul arguments must both be tree_math.MatrixMixin or tree_math.VectorMixin objects",
    ):
      matrix1 @ jnp.ones((9,))  # pylint: disable=expression-not-assigned

  def test_custom_class(self):

    @tree_util.register_pytree_node_class
    class CustomMatrix(tm.MatrixMixin):

      def __init__(self, a: int, b: float, c: int, d: float):
        self.row1 = [a, b]
        self.row2 = [c, d]
        self.outer_treedef = [leaf_pytree, leaf_pytree]
        self.outer_ndims = [0, 0]
        self.trans = False


      def tree_flatten(self):
        children = (self.row1, self.row2)
        aux_data = (self.outer_treedefs, self.outer_ndims, self.trans)
        return (children, aux_data)

      @classmethod
      def tree_unflatten(cls, _, args):
        (a, b), (c, d) = children
        return cls(a, b, c, d)

    v1 = CustomMatrix(1, 2, 3, 4.)
    v2 = v1 + 3
    self.assertTreeEqual(v2, CustomMatrix(4, 5, 6, 7.), check_dtypes=True)
    v3 = v1 + v2
    self.assertTreeEqual(v3, CustomMatrix(5, 7, 9, 11.), check_dtypes=True)
    v4 = v1 @ v2
    self.assertTreeEqual(v4, CustomMatrix(16, 19., 36., 43.), check_dtypes=True)

if __name__ == "__main__":
  absltest.main()
